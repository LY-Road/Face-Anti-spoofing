# Face Anti-spoofing Attack Detection
● 以FaceBagNet模型为主线，通过二分类实现真实人脸和攻击人脸的检测。

### 下载CASIA-SURF数据集
链接：https://pan.baidu.com/s/1o3E_TWAoVENMWAMqZr5csw 提取码d55a
将数据集放在data文件夹下。

### 下载预训练权重
单模态color (model_A)：
三模态fusion(model_A)：

### 训练
#### 1）单模态
train model_A with color imgs， patch size 48：  
> $ CUDA_VISIBLE_DEVICES=0 python3 train_CyclicLR.py --model=model_A --image_mode=color --image_size=48  
#### 2）三模态 
train model A fusion model with multi-modal imgs， patch size 48：  
> $ CUDA_VISIBLE_DEVICES=0 python3 train_Fusion_CyclicLR.py --model=model_A --image_size=48  

### 测试
#### 1）单模态
infer model_A with color imgs， patch size 48：  
> $ CUDA_VISIBLE_DEVICES=0 python3 train_CyclicLR.py --mode=infer_test --model=model_A --image_mode=color --image_size=48  
#### 2) 三模态
infer model A fusion model with multi-modal imgs， patch size 48： 
> $ CUDA_VISIBLE_DEVICES=0 python train_Fusion_CyclicLR.py --mode=infer_test --model=model_A --image_size=48

