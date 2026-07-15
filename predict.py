import os
import argparse
import pandas as pd
import joblib
from features import extract_features_for_pause

def main():
    parser = argparse.ArgumentParser(description="EOT Prediction Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data folder containing labels.csv")
    parser.add_argument("--out", type=str, required=True, help="Path to the output predictions.csv file")
    args = parser.parse_args()

    # Resolve model path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "eot_model.joblib")
    
    print(f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Read strictly required columns, ignore pause_end and label entirely
    csv_path = os.path.join(args.data_dir, "labels.csv")
    df = pd.read_csv(csv_path, usecols=["turn_id", "audio_file", "pause_index", "pause_start"])

    predictions = []
    
    print(f"Extracting features and predicting for {len(df)} pauses...")
    for _, row in df.iterrows():
        audio_path = os.path.join(args.data_dir, row["audio_file"])
        
        # Extract features using strict causal windowing
        feats = extract_features_for_pause(audio_path, row["pause_start"], row["pause_index"])
        
        # Enforce column order to match training exactly
        X = pd.DataFrame([feats])
        X = X[feature_columns] 
        
        # Predict probability of EOT
        p_eot = model.predict_proba(X)[0, 1]
        predictions.append({
            "turn_id": row["turn_id"],
            "pause_index": row["pause_index"],
            "p_eot": p_eot
        })

    out_df = pd.DataFrame(predictions)
    out_df.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()