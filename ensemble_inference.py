import os
import argparse
import pandas as pd
import numpy as np
import torch
from config import Config
# Import the existing inference function (ensuring it uses 5-View TTA)
from inference_pseudo import inference_per_model 

# --- Champion Settings ---
# Voting Weights are now defined in config.py

def optimize_predictions(probs):
    """
    【Post-Processing】Threshold Optimization for Rare Class (Label 2: Horizontal Defect)
    """
    final_preds = np.argmax(probs, axis=1)
    
    # Aggressive threshold for Label 2
    THRESHOLD_LABEL_2 = 0.4
    
    label_2_candidates = probs[:, 2] > THRESHOLD_LABEL_2
    
    original_count = (final_preds == 2).sum()
    final_preds[label_2_candidates] = 2
    new_count = (final_preds == 2).sum()
    
    if new_count > original_count:
        print(f"⚡ Threshold Optimization: Rescued {new_count - original_count} samples as Label 2!")
    
    return final_preds

def run_multi_model_ensemble(model_weights, output_name='submission_ensemble.csv', debug=False):
    print(f"🚀 Starting Multi-Model Champion Ensemble...")
    print(f"🎯 Target Weights: {model_weights}")
    
    # Load Test Data
    test_df = pd.read_csv(Config.TEST_CSV)
    if debug:
        test_df = test_df.sample(50).reset_index(drop=True)
        print("⚠️ DEBUG MODE: Using only 50 samples.")
    
    final_ensemble_probs = None
    total_weight_used = 0.0
    
    # --- Loop through each Architecture ---
    # Sequential Mode: Load one architecture -> Predict -> Delete -> Load next
    for model_name, weight in model_weights.items():
        if model_name not in Config.MODELS:
            print(f"⚠️ Model {model_name} not found in Config.MODELS, skipping.")
            continue
            
        # Check if model files actually exist to avoid wasting time
        # (This check is rough, inference_per_model does a better check per fold)
        
        print(f"\n[Ensemble Step] Processing {model_name.upper()} (Weight: {weight})...")
        
        # Call inference_per_model (Handles 5-Fold + TTA internally)
        probs = inference_per_model(model_name, test_df, Config.DEVICE)
        
        if probs is not None:
            # Weighted Accumulation
            if final_ensemble_probs is None:
                final_ensemble_probs = probs * weight
            else:
                final_ensemble_probs += probs * weight
            
            total_weight_used += weight
            print(f"✅ {model_name} added to ensemble.")
        else:
            print(f"❌ Warning: Prediction failed for {model_name}. Skipping.")

    if final_ensemble_probs is None:
        print("❌ Error: No models ran successfully.")
        return

    # --- Normalization ---
    # Divide by total weight to get valid probabilities
    print(f"\nNormalization: Dividing by total weight {total_weight_used:.2f}")
    final_ensemble_probs /= total_weight_used

    # --- Post-Processing Magic ---
    print("\n[Post-Processing] Applying Threshold Optimization...")
    preds = optimize_predictions(final_ensemble_probs)
    
    # --- Save Submission ---
    submission_df = test_df.copy()
    submission_df['Label'] = preds
    
    if 'ID' in submission_df.columns:
        submission_df = submission_df[['ID', 'Label']]
    
    output_path = os.path.join(Config.SUBMISSION_DIR, output_name)
    submission_df.to_csv(output_path, index=False)
    
    print(f"✅ Ensemble Submission Saved: {output_path}")
    
    # Check Distribution
    print("\nFinal Prediction Distribution:")
    print(submission_df['Label'].value_counts().sort_index())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Specific model to run (e.g., convnext). If None, runs multi-model ensemble.')
    parser.add_argument('--debug', action='store_true', help='Run on small subset')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output filename')
    args = parser.parse_args()
    
    # [Dual Mode Logic]
    # If user specifies --model, run Single Model Champion Mode
    # Else, run Multi-Model Ensemble Mode using predefined weights
    if args.model:
        print(f"🔹 Single Model Mode Selected: {args.model}")
        # Override weights to focus 100% on the single model
        weights = {args.model: 1.0}
    else:
        print(f"🔹 Multi-Model Ensemble Mode Selected")
        # Use Config weights
        weights = Config.MODEL_WEIGHTS
    
    run_multi_model_ensemble(model_weights=weights, output_name=args.output, debug=args.debug)
