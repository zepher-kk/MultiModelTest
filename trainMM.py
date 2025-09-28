#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11.yaml')
    # model = YOLOMM('yolo11n-mm-mid3.yaml')
    model.train(data='D:\learn\dmt\MM\M3FD_yolo_10percent\data.yaml',
                epochs=100,batch=8,
                # modality='X',
                cache=True,exist_ok=True,
                project='ResTest',name='Test/YOLOMM')