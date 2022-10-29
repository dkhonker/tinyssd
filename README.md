## 环境配置

pytorch 1.7.0

CUDA 11.0

GTX 2080Ti

## 训练配置

batch size 为64

epoch为50

学习率为0.2

weight decay 为5e-4

## 模型中使用的方法

### 原始架构

原始架构来自老师给的Tinyssd代码。

### VGG结构

VGG结构由Visual Geometry Group提出，是一种使用多个3 * 3卷积叠加的模型。

我们将其替换原始Tinyssd网络中的base_net。

### Resnet9架构

Resnet9是一种由David Page提出的，在CIFAR10数据集上在79秒内达到94%准确率的模型。是DAWNBench排行榜上花费最短的训练时间在CIFAR10上达到94%准确率的模型。

我们将其替换原始Tinyssd网络中的base_net。

### CBAM

CBAM( Convolutional Block Attention Module )是一种轻量级注意力模块的提出于2018年,它可以在空间维度和通道维度上进行Attention操作。

![image-20221029205719061](C:\Users\李\AppData\Roaming\Typora\typora-user-images\image-20221029205719061.png)

## 代码使用方法

参数：

- **mode**-  模式，可选`train`或`test`.默认 `train`

- **epochs**-  默认 `3`
- **batch_size** - 默认`64`
- **lr** - 学习率. 默认 `0.2`
- **weight_decay** -  默认 `5e-4`
- **wandb_use** - 是否使用wandb来记录训练过程。**注意！**若需使用需要先在`solver/train.py`代码中填入自己的key，默认 `False`
- **seed** - 随机种子. 默认 `3607`
- **pretrain** - 需要加载的模型权重，用于预训练或者测试。默认 `None`
- **backbone** - 可选`base`(即原始架构)、`vgg` 、`resnet` .默认`base`
- **CBAM** - 是否使用CBAM模块。默认`False`

示例：

- 训练

（不用wandb时）

`python main.py --epochs 50  --mode train  --backbone resnet --CBAM True`

（需要用wandb时）

`python main.py --epochs 50  --mode train  --backbone resnet --CBAM True --wandb_use True`

- 测试

`python main.py  --mode test --pretrain weights/resnet+CBAM/net_50.pkl --backbone resnet --CBAM True`

## 代码文件说明

  ```
  tinyssd/
  │
  ├── main.py - 主函数
  │
  ├── config.py - 获取执行参数
  │
  ├── dataset/ - 数据集，分为train/test/val
  │
  ├── data/ - 读取数据集
  │   ├── dataset.py
  │   └── loader.py
  │
  ├── weights/ - 存储模型权重
  │
  ├── weights/ - 存储测试图像
  │
  ├── models/ - 模型文件
  │   ├── CBAM.py 
  │   ├── VGG.py 
  │   ├── resnet.py
  │   └── model.py - 主模型
  ├── solver/ - 包含训练和测试
  │   ├── train.py - 根据传入的参数训练网络
  │   ├── test.py - 测试并保持检测结果
  │   ├── loss.py - 计算loss用于训练
  │   └── trainer.py
  │
  └── utils/ 
      ├── model.py - 计算模型参数量
      └── utils.py - 一些比较杂的函数
  ```

## 实验结果

| 模型架构    | 效果图                                                       |
| ----------- | ------------------------------------------------------------ |
| 原始        | ![base](demo\base.jpg) |
| 原始+CBAM   | ![base](demo\base+CBAM.jpg) |
| resnet结构  | ![base](demo\resnet.jpg) |
| resnet+CBAM | ![base](demo\resnet+CBAM.jpg) |
| VGG         | ![base](demo\vgg.jpg) |
| VGG+CBAM    | ![base](demo\vgg+CBAM.jpg) |

参考资料：

1.DAWNBench: https://dawn.cs.stanford.edu/benchmark/index.html#cifar10

2.CBAM:https://arxiv.org/abs/1807.06521
