# scripts/annotation.py
import os
import json
from utils.config import CHARTS_DIR, YOLO_CLASSES

def convert_annotations(input_dir, output_dir, classes):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            with open(os.path.join(input_dir, file), 'r') as f:
                data = json.load(f)
            yolo_annotations = []
            for obj in data['shapes']:
                class_name = obj['label']
                if class_name not in classes:
                    continue
                class_id = classes.index(class_name)
                points = obj['points']
                x_min = min([p[0] for p in points])
                y_min = min([p[1] for p in points])
                x_max = max([p[0] for p in points])
                y_max = max([p[1] for p in points])
                width = x_max - x_min
                height = y_max - y_min
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                # Normalize coordinates
                img_width = data['imageWidth']
                img_height = data['imageHeight']
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            # Save YOLO annotations
            image_name = file.replace('.json', '.png')
            with open(os.path.join(output_dir, image_name.replace('.png', '.txt')), 'w') as f:
                f.write('\n'.join(yolo_annotations))

def main():
    input_dir = os.path.join(CHARTS_DIR, "annotations")  # Folder where JSON annotations are saved
    output_dir = os.path.join(CHARTS_DIR, "yolo_annotations")
    convert_annotations(input_dir, output_dir, YOLO_CLASSES)
    print("Annotation conversion completed.")

if __name__ == "__main__":
    main()
