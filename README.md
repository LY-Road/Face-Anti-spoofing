# Face Anti-spoofing Attack Detection
● 以FaceBagNet模型为主线，通过二分类实现真实人脸和攻击人脸的检测。

500个epochs：
mode|model|acer
---|--|--
color|model_A|0.0666
fusion|model_A|0.0025

### 下载CASIA-SURF数据集
链接：https://pan.baidu.com/s/1o3E_TWAoVENMWAMqZr5csw  提取码：d55a  

### 修改变量
修改process/data_helper中关于数据地址的变量。

### 训练
#### 1）单模态
train model_A with color imgs， patch size 48：  
> $ CUDA_VISIBLE_DEVICES=0 python3 train_CyclicLR.py --model=model_A --image_mode=color --image_size=48  

可以通过修改--model更换模型，可以通过修改--image_mode更换模态。
#### 2）三模态 
train model A fusion model with multi-modal imgs， patch size 48：  
> $ CUDA_VISIBLE_DEVICES=0 python3 train_Fusion_CyclicLR.py --model=model_A --image_size=48  

可以通过修改--model更换模型。

### 测试
#### 1）单模态
infer model_A with color imgs， patch size 48：  
> $ CUDA_VISIBLE_DEVICES=0 python3 train_CyclicLR.py --mode=infer_test --model=model_A --image_mode=color --image_size=48  


可以通过修改--model更换模型，可以通过修改--image_mode更换模态。
#### 2) 三模态
infer model A fusion model with multi-modal imgs， patch size 48： 
> $ CUDA_VISIBLE_DEVICES=0 python3 train_Fusion_CyclicLR.py --mode=infer_test --model=model_A --image_size=48  

可以通过修改--model更换模型。

### 其他解释
 -  单模态的facebagnet的训练代码: train_CyclicLR.py   
 -  三模态的facebagnet的训练代码: train_Fusion_CyclicLR.py  
 -  训练命令样例：run.sh
 -  数据准备代码所在文件夹：process  
 -  模型准备代码所在文件夹：model,model_fusion   
 -  输出log文件所在文件夹：models  
 -  指标计算代码：metric.py,submission.py  
 -  其他有用函数收集文件：utils.py  

