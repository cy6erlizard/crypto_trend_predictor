# scripts/pattern_sequence_analysis.py
import os
import pandas as pd
from ultralytics import YOLO
from utils.config import YOLO_MODEL_PATH, CHARTS_DIR, YOLO_CLASSES

def detect_patterns(chart_image_path, model):
    results = model.predict(chart_image_path)
    patterns = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = YOLO_CLASSES[cls_id]
            # Assuming bounding box contains timestamp info; otherwise, you need to map it
            # Here, we simplify by storing the pattern name
            patterns.append({'pattern': class_name, 'chart': chart_image_path})
    return patterns

def main():
    model = YOLO(YOLO_MODEL_PATH)
    charts = [os.path.join(CHARTS_DIR, f) for f in os.listdir(CHARTS_DIR) if f.endswith('.png')]
    all_patterns = []
    for chart in charts:
        patterns = detect_patterns(chart, model)
        all_patterns.extend(patterns)
        print(f"Detected {len(patterns)} patterns in {chart}")
    
    df = pd.DataFrame(all_patterns)
    df.to_csv(os.path.join(CHARTS_DIR, 'detected_patterns.csv'), index=False)
    print("Pattern detection completed and saved to detected_patterns.csv")
    
    # Analyze sequences of three consecutive patterns
    df_sorted = df.sort_values(by='chart')  # Ensure chronological order
    df_sorted['pattern_sequence'] = df_sorted['pattern'].rolling(window=3).apply(lambda x: '->'.join(x), raw=True)
    df_sorted.dropna(inplace=True)
    
    # Define bullish and bearish sequences
    bullish_sequences = [
        "inverse_head_and_shoulders->ascending_triangle->flag",
        # Add more bullish sequences as needed
    ]
    bearish_sequences = [
        "head_and_shoulders->descending_triangle->flag",
        # Add more bearish sequences as needed
    ]
    
    df_sorted['trend'] = df_sorted['pattern_sequence'].apply(
        lambda x: 'bullish' if x in bullish_sequences else ('bearish' if x in bearish_sequences else 'neutral')
    )
    
    confirmed_trends = df_sorted[df_sorted['trend'] != 'neutral']
    confirmed_trends.to_csv(os.path.join(CHARTS_DIR, 'confirmed_trends.csv'), index=False)
    print("Trend confirmation completed and saved to confirmed_trends.csv")

if __name__ == "__main__":
    main()
