#这是热力图可视化的方法，请你参考我的使用
#在使用的时候，双模态，单模态推理都是列表输入，双模态输入强制第一个输入为RGGB，第二个为X
#单模态输入，请使用modality='rgb'或者modality='X'来指定输入模态
from ultralytics import YOLOMM

model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM/weights/best.pt')
model.vis(source=['/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
                  # '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png'
                  ],
          method='heatmap',  #热力图
          layers=[2,4,6,18,28],  # 使用yaml层
          modality='rgb',
          alg='gradcam',
          save=True,
          project='ResTest/vis',
          name='Vis')