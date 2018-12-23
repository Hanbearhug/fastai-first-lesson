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
