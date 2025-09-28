
from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11.yaml')
    # model = YOLOMM('yolo11n-mm-mid3.yaml')
    model.train(data='D:\learn\dmt\MM\M3FD_yolo_10percent\data.yaml',
                epochs=100,batch=8,
                # modality='X',
                cache=True,exist_ok=True,
                project='ResTest',name='Test/YOLOMM')
