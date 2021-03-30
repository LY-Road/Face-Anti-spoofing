import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
import sys
sys.path.append("..")
from process.data_fusion import *
from process.augmentation import *
from metric import *
from collections import OrderedDict

pwd = os.path.abspath('./')

def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet
    net = FusionNet(num_class=num_class)
    return net

def load_test_list():
    list = []
    # f = open(DATA_ROOT + '/test_public_list.txt')
    f = open(DATA_ROOT + '/val_private_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

model_path = os.path.join(pwd, 'models', 'model_A_fusion_64', 'checkpoint', 'global_min_acer_model.pth')
net = get_model(model_name='model_A', num_class=2)
if torch.cuda.is_available():
    state_dict = torch.load(model_path, map_location='cuda')
else:
    state_dict = torch.load(model_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
# net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print('loaded model from: ', model_path)
if torch.cuda.is_available():
    net = net.cuda()

index = 7
test_list = load_test_list()
print(test_list[index])
color,depth,ir = test_list[index][:3]
test_id = color+' '+depth+' '+ir
color = cv2.imread(os.path.join(DATA_ROOT, color),1)
depth = cv2.imread(os.path.join(DATA_ROOT, depth),1)
ir = cv2.imread(os.path.join(DATA_ROOT, ir),1)
# whole_image = np.concatenate([color, depth, ir], axis=0)
cv2.imshow('color', color)
cv2.imshow('depth', depth)
cv2.imshow('ir', ir)
cv2.waitKey(0)
cv2.destroyWindow('whole_image')

color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))
color = color_augumentor(color, target_shape=(64, 64, 3), is_infer=True)
depth = depth_augumentor(depth, target_shape=(64, 64, 3), is_infer=True)
ir = ir_augumentor(ir, target_shape=(64, 64, 3), is_infer=True)
n = len(color)
color = np.concatenate(color, axis=0)
depth = np.concatenate(depth, axis=0)
ir = np.concatenate(ir, axis=0)

image = np.concatenate([color.reshape([n, 64, 64, 3]),
                        depth.reshape([n, 64, 64, 3]),
                        ir.reshape([n, 64, 64, 3])],
                       axis=3)
image = np.transpose(image, (0, 3, 1, 2))
image = image.astype(np.float32)
image = image.reshape([n, 3 * 3, 64, 64])
image = image / 255.0
input_image = torch.FloatTensor(image)
if (len(input_image.size())==4) and torch.cuda.is_available():
    input_image = input_image.unsqueeze(0).cuda()
elif (len(input_image.size())==4) and not torch.cuda.is_available():
    input_image = input_image.unsqueeze(0)
label = test_id
# return torch.FloatTensor(image), test_id

b, n, c, w, h = input_image.size()
input_image = input_image.view(b*n, c, w, h)
if torch.cuda.is_available():
    input_image = input_image.cuda()

with torch.no_grad():
    logit,_,_   = net(input_image)
    logit = logit.view(b,n,2)
    logit = torch.mean(logit, dim = 1, keepdim = False)
    prob = F.softmax(logit, 1)

print('probabilistic：', prob)
try:
    print('label: ', test_list[index][-1], '，predict: ', np.argmax(prob.detach().cpu().numpy()))
except:
    print('predict: ', np.argmax(prob.detach().cpu().numpy()))
