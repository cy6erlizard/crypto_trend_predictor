# scripts/train_object_detection.py
import os
from ultralytics import YOLO
from utils.config import YOLO_CLASSES, CHARTS_DIR, OBJECT_DETECTION_MODEL_DIR

def prepare_data_yaml():
    data_yaml = {
        'train': os.path.join(CHARTS_DIR, 'yolo_annotations', 'train'),
        'val': os.path.join(CHARTS_DIR, 'yolo_annotations', 'val'),
        'nc': len(YOLO_CLASSES),
        'names': YOLO_CLASSES
    }
    with open(os.path.join(OBJECT_DETECTION_MODEL_DIR, 'data.yaml'), 'w') as f:
        import yaml
        yaml.dump(data_yaml, f)
    print("data.yaml created.")

def train_yolo():
    model = YOLO('yolov8n.pt')  # Using YOLOv8 Nano as base
    model.train(data=os.path.join(OBJECT_DETECTION_MODEL_DIR, 'data.yaml'),
                epochs=50,
                imgsz=640,
                batch=16,
                name='yolov8_crypto_patterns',
                project=OBJECT_DETECTION_MODEL_DIR)

def main():
    os.makedirs(OBJECT_DETECTION_MODEL_DIR, exist_ok=True)
    prepare_data_yaml()
    train_yolo()

if __name__ == "__main__":
    main()
