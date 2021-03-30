import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '6'
import sys
sys.path.append("..")
sys.path.append("./process")
sys.path.append("./model")
import argparse
from process.data_fusion import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart


def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet
    elif model_name == 'resnet18_facebagnet':
        from model_fusion.model_baseline_Fusion import FusionNet

    net = FusionNet(num_class=num_class)
    return net

def run_train(config):
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    criterion          = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'),mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = 8)

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size // 36,
                                drop_last   = False,
                                num_workers = 8)

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, num_class=2)
    print(net)
    net = torch.nn.DataParallel(net)
    #net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('criterion=%s\n'%criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    train_loss   = np.zeros(6,np.float32)
    valid_loss   = np.zeros(6,np.float32)
    batch_loss   = np.zeros(6,np.float32)
    iter = 0
    i    = 0

    start = timer()
    #-----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    global_min_acer = 1.0

    # cycle_num = 10：10个周期
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        # cycle_inter = 50：在50个epoch内，学习率从最大值降到最小值 （一个周期）
        for epoch in range(0, config.cycle_inter):
            # 调整学习率（周期余弦退火）
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()

            # 一个epoch
            for input, truth in train_loader:
                iter = i + start_iter
                # one iteration update  -------------
                net.train()
                input = input.cuda()
                truth = truth.cuda()

                logit,_,_ = net.forward(input)
                truth = truth.view(logit.shape[0])

                loss  = criterion(logit, truth)
                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array(( loss.item(), precision.item(),))
                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum = 0

                i = i + 1

            # 当epoch大于25的时候：eval
            if epoch >= config.cycle_inter // 2:
            # if 1:
                net.eval()
                valid_loss, _ = do_valid_test(net, valid_loader, criterion)
                net.train()

                # 小于当前循环中最小的acer，保存模型
                if valid_loss[1] < min_acer and epoch > 0:
                    min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                # 小于总体最小的acer，保存模型
                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')

            asterisk = ' '
            log.write(config.model_name+' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start), 'min')))

        ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')

def run_test(config, dir):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = './models'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2)
    net = torch.nn.DataParallel(net)
    #net =  net.cuda()

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    test_dataset = FDDataset(mode = 'test', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    test_loader  = DataLoader( test_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    criterion = softmax_cross_entropy_criterion
    net.eval()

    valid_loss,out = do_valid_test(net, valid_loader, criterion)
    print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

    print('infer!')
    out = infer_test(net, test_loader)
    print('done')

    submission(out,save_dir+'_noTTA.txt', mode='test')

def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_mode', type=str, default='fusion')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    config = parser.parse_args()
    print(config)
    main(config)
