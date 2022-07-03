# MindSpore-Misson-5

上一个阶段使用ModelArts训练了一个蘑菇分类器，该分类器可以对9种蘑菇进行分类。

现在，请使用上一个阶段训练好的网络作为本作业的backbone网络，训练一个可以分类另外2种新蘑菇的分类器。

## 数据集

本作业使用kaggle的另一个蘑菇数据集，只需对其中的两种新蘑菇进行分类：

-   Exidia：黑耳，又称为黑胶菌，分布于我国吉林、河北、山西、宁夏、青海、四川、广西、海南,西藏等地，无毒。

-   Inocybe：丝盖伞，又称毛丝盖菌、毛锈伞，分布于我国河北、吉林、江苏、山西、四川、云南、甘肃、新疆等地，有毒。


## 环境

使用云主机的Ascend计算

## 特征提取

下面是9蘑菇分类的实验截图。保留其学习到的特征：best\_param\_backbone.ckpt

<img src="./attachments/media/image3.png" style="width:2.00069in;height:1.93333in" /><img src="./attachments/media/image4.png" style="width:3.70278in;height:1.925in" /><img src="./attachments/media/image5.png" style="width:1.94722in;height:1.88194in" /><img src="./attachments/media/image6.png" style="width:3.75417in;height:1.90139in"/>

选择网络、选择损失函数、优化器、模型。

## 训练

```
model.train(config.epochs, train\_data, callbacks=cb)
```

<img src="./attachments/media/image7.png" style="width:6.69028in;height:6.07986in" />

## 模型评估

### 可视化loss

<img src="./attachments/media/image8.png" style="width:6.68403in;height:5.8125in" />

### 可视化accuracy

<img src="./attachments/media/image9.png" style="width:6.68542in;height:5.29028in" />

### 测试效果

accuracy 约 0.94

<img src="./attachments/media/image10.png" style="width:6.67847in;height:1.21667in" />

## 效果展示

<img src="./attachments/media/image11.png" style="width:6.69097in;height:3.50833in" />

