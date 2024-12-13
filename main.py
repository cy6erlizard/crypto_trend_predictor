# main.py
import argparse
from scripts import (
    data_collection,
    chart_generation,
    annotation,
    train_object_detection,
    pattern_sequence_analysis,
    predictive_modeling,
    trading
)

def main(args):
    if args.step == "collect_data":
        data_collection.main()
    elif args.step == "generate_charts":
        chart_generation.main()
    elif args.step == "convert_annotations":
        annotation.main()
    elif args.step == "train_yolo":
        train_object_detection.main()
    elif args.step == "detect_patterns":
        pattern_sequence_analysis.main()
    elif args.step == "train_predictive_model":
        predictive_modeling.main()
    elif args.step == "execute_trading":
        trading.main()
    else:
        print("Unknown step.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Trend Predictor Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        help="Step to execute: collect_data, generate_charts, convert_annotations, train_yolo, detect_patterns, train_predictive_model, execute_trading")
    args = parser.parse_args()
    main(args)
