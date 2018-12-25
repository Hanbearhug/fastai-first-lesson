# fastai-first-lesson
## 安装fastai0.7.0
安装过程要吐一口老血，采坑无数，下面记录成功的过程
```
git clone https://github.com/fastai/fastai.git
cd fastai
conda env create -f environment.yml
```
注意gitbash默认会把fastai下载到c盘。启动时将fastai复制过去就行(或者直接改了gitbash的默认路径)，然后就是进入fastai的目录下配置环境。
```
cd courses\ml1
del fastai
mklink /d fastai ..\..\old\fastai
cd ..\..
```
右键打开anaconda界面运行上面一段神秘代码。。
```
conda activate fastai
```
每次进入anaconda界面激活fastai，然后启动jupyter notebook

```
import sys
sys.path.append('home/zzz/fastai-master/old')  #注意这里添加的是fastai的git包中的old文件夹路径，这个才对应的是fastai0.7
from fastai.conv_learner import *
```
进入notebook后要记得在路径中加入old的文件见，这里面的才是fastai0.7.0(求求大佬开院支持windows的fastai以及课程叭！)

## 开始学习第一课
```
img = plt.imread(f'{PATH}valid/cats/files[0]')
plt.imshow(img)
```
plt.imread用于读取一个image文件，f-string语法用于字符串的格式化(优于%-string)

### 三行代码开始模型
```
arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01,2)
```
这里ImageClassifierData.from_paths用于第一种标准数据集的训练及预测，tfms_from_model用于对模型进行类似于数据增强的操作。


### 选择学习率
学习率是最难设定的参数之一，它显著的影响了模型的表现
```
learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf = learn.lr_find()
```
learn.lr_find()用于帮助寻找最优的学习率，其思想来源于2015年的论文"Cyclical Learning Rates for Training Neural Networks"，做法是先从很小的学习率开始，然后逐步提高学习率(一般是两倍)直到损失函数停止下降为止。
```
learn.sched.plot_lr()
```
![](https://github.com/Hanbearhug/fastai-first-lesson/blob/master/%E8%BF%AD%E4%BB%A3%E5%AD%A6%E4%B9%A0%E7%8E%87%E5%9B%BE%E5%83%8F.png)
```
learn.sched.plot()
```
![](https://github.com/Hanbearhug/fastai-first-lesson/blob/master/%E5%AD%A6%E4%B9%A0%E7%8E%87%E6%8D%9F%E5%A4%B1%E5%9B%BE%E5%83%8F.png)
第一张图是学习率随迭代次数的变化图像，第二张是损失函数随学习率变化的图像，一般而言选择损失函数最低点的前一个数量级作为最优学习率，原因其一是因为当损失函数到达最低点时，事实上学习率已经偏大，同时又要在SGD with restarts中保持较大的重启学习率以使得参数可以跳出比较狭窄的地带到达比较平缓的地带从而提升泛化能力。

### 数据增强
当数据开始过拟合时，我们一般需要采取数据增强的手段去缓解这种问题，避免模型学习了一些训练集上特定的特征，数据增强需要针对不同的数据集进行设置，例如，当观看猫狗的图片时，我们一般采用左右移动、翻转、小角度旋转等策略，原因在于一般猫狗图片不会上下颠倒，而观看卫星图像时，由于不同的拍摄角度问题，也许就需要考虑上下颠倒图片，而手写体识别则一般不要求翻转，原因是特定的方向决定了手写体的含义。
```
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
```
这里使用transforms_side_on进行水平翻转操作，而使用max_zoom来进行局部放大操作，效果如下所示
![](https://github.com/Hanbearhug/fastai-first-lesson/blob/master/%E6%B0%B4%E5%B9%B3%E7%BF%BB%E8%BD%AC.png)

```
learn.fit(1e-2, 3, cycle_len=1)
```
这里的cycle_len代表着学习率重启的周期，即每cycle_len个epoch周期进行一次学习率的重启(SGD with restarts)
![](https://github.com/Hanbearhug/fastai-first-lesson/blob/master/SGD_with_starts.png)

### Fine-tuning and differential learning rate annealing
在调完最后一层后我们开始调节其他预训练过的层，由于初始的层已经能够很好的学习一些初等的特征了，我们不希望去破坏这些已经训练好的权重，因此我们采用差异化的学习率的方式来进行训练，这里要注意的是，如果说图像与ImageNet的图像数据集比较类似，那么我们给前面的层设定的学习率就要小些，因为我们更希望它们的参数不要波动太大，但是如果图像与ImageNet的图像相差较远(例如卫星图像)，我们一般就要采取更大的学习率。
```
learn.unfreeze()
lr = np.array([1e-4, 1e-3, 1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
```
这里的cycle_mult参数代表着每一次cycle结束后新的cycle变为原先的cycle的几倍，这是为了防止cycle太小而产生的欠拟合现象。
