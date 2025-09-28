# 多模态目标检测（可见光+红外）

## 概述

针对输入为可见光图像(vis)与红外图像(ir)进行目标检测**精度，速度**的研究

## 方法

多模态目标检测中有4种核心的融合策略：**串联融合（Series Fusion）**， **前期融合（Early Fusion）**， **中期融合（Mid Fusion）**和 **后期融合（Late Fusion）**。这四种策略的核心区别在于**融合发生在了网络处理流程的哪个阶段**。

### 串联融合（Series Fusion）



![串联融合](D:\hzx\mission\diaoyan\img\img1.png)

串联融合，也称像素级融合，指在输入到目标检测网络**之前**，先将来自不同传感器（这里是可见光和红外）的原始图像数据在像素层面上合并成一幅**全新的、融合后的图像**。然后将这幅融合图像送入一个**单一的标准目标检测网络**（如YOLO、Faster R-CNN）中进行检测。本研究是使用[ICAFusion](#[GitHub - chanchanchan97/ICAFusion: ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection, Pattern Recognition](https://github.com/chanchanchan97/ICAFusion))进行像素级融合。

### **前期融合（Early Fusion）**， **中期融合（Mid Fusion）**和 **后期融合（Late Fusion）**



这三个融合方式区别于串联融合是：不通过额外的融合步骤，直接输入到目标检测网络，通过深度学习的反向传播学习各个通道之间的权重。

**前期融合**如下所示，即将原来用于可见光目标检测模型通道扩充（c=3 --> c=6）, 依托通用目标检测的框架，依托卷积(cnn)进行通道融合。

![前期融合](D:\hzx\mission\diaoyan\img\img2.png)

**中期融合**如下所示，即各个模态通过不同的特征采样后融合后通过颈部网络(neck)与检测头(detect)。

![中期融合](D:\hzx\mission\diaoyan\img\img3.png)

**后期融合**（多尺度融合）如下所示，与中期融合相似，不同的是不同模态以多尺度方式进行融合。

![后期融合](D:\hzx\mission\diaoyan\img\img4.png)

## 实验

### 实验介绍

本实验基于M3FD数据集（可见光+红外），采用原数据集10%的数据（包含294张train，84张val，42张test）进行训练、推理。实验设备为I9-14900K，RTX5090。torch版本为2.8。

串联融合的融合方式采用ICAFusion，检测网络使用YOLOv11n及其改进网络。

<center class="half">
    <img src="D:\hzx\mission\diaoyan\img\vis.png" width="400"/>
    <img src="D:\hzx\mission\diaoyan\img\ir.png" width="400"/>
</center>

### 实验数据

| 融合方式 | AP    | AP50  | preprocess(ms) | inference(ms) | *postprocess(ms) | all(ms) |
| -------- | ----- | ----- | -------------- | ------------- | ---------------- | ------- |
| 串联融合 | 0.401 | 0.248 | *24+0.8        | 13.3          | 41.5             | 79.6    |
| 前期融合 | 0.52  | 0.319 | *3.1           | 14            | 37.7             | 54.8    |
| 中期融合 | 0.52  | 0.319 | 2.7            | 20.2          | 62.9             | 85.8    |
| 后期融合 | 0.477 | 0.298 | 2.7            | 17.7          | 69               | 89.4    |

* *24 指通过ICAFusion的时间。统计了图片保存的耗时，无需保存的情况下会短一些

* *3.1 指图片进入网络的前处理时间，其中额外包括对两张输入图片进行通道堆叠的耗时

  * 请注意：4个融合方式输入分别为三通道，六通道，两张三通道（vis&ir），两张三通道(同上)

* *postprocess指后处理，主要包括NMS阈值筛选、绘制目标框和图片输出保存过程。理论上可以再缩减时间。

  ### 结果

  <center class="half">
      <img src="D:\hzx\mission\diaoyan\img\00004_fused.jpg" width="400"/>
      <img src="D:\hzx\mission\diaoyan\img\early\image0.jpg" width="400"/>
  </center>

​                                                                                 串联融合                                                                                    前期融合



![后期融合](D:\hzx\mission\diaoyan\img\mid\00004_multimodal.jpg)

​																中期融合

![后期融合](D:\hzx\mission\diaoyan\img\late\00004_multimodal.jpg)

​															   后期融合





## 结论

在少数据集情况下，以YOLOv11n为baseline，按照精度与实时性进行评估，前期融合(c=6)最优。



参考：[MultiModelTest](#git@github.com:zepher-kk/MultiModelTest.git)

