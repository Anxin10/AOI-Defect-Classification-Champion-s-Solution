import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import gc

from config import Config
from dataset import AOIDataset, get_transforms

def tta_inference(model, images):
    """
    🚀 Parallel TTA (5 Views):
    Stack 5 views to feed GPU at once -> Maximize VRAM usage & Speed.
    """
    batch_size = images.size(0)
    
    # 1. Prepare 5 Views on GPU
    views = [
        images,                                       # Original
        torch.flip(images, dims=[3]),                 # H-Flip
        torch.flip(images, dims=[2]),                 # V-Flip
        torch.rot90(images, k=1, dims=[2, 3]),        # Rot 90
        torch.rot90(images, k=3, dims=[2, 3])         # Rot 270
    ]
    
    # 2. Stack into Huge Batch [Batch * 5, C, H, W]
    stacked_images = torch.cat(views, dim=0)
    
    # 3. Parallel Inference
    logits = model(stacked_images)
    probs = F.softmax(logits, dim=1)
    
    # 4. Reshape and Average
    # [Batch * 5, Num_Classes] -> [5, Batch, Num_Classes]
    probs = probs.view(5, batch_size, -1)
    return probs.mean(dim=0)

@torch.no_grad()
def inference(models_dict, test_loader, device):
    pass 

import gc 

def inference_per_model(model_name, test_df, device):
    torch.set_float32_matmul_precision('high')    
    
    img_size = Config.IMG_SIZES[model_name]
    timm_name = Config.MODELS[model_name]
    
    test_dataset = AOIDataset(
        test_df, 
        transforms=get_transforms(img_size, mode='test'),
        mode='test'
    )
    
    # ⚠️ [Optimization] Reduce Workers to save RAM, Parallel TTA to boost GPU
    target_workers = 2  * 4
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE  * 4  , # No multiplier, since Parallel TTA expands x5
        shuffle=False, 
        num_workers=target_workers,   # Force 2 workers
        pin_memory=True
    )
    
    # Store predictions from all folds for Power Mean
    all_fold_preds = [] 
    
    print(f"Running Inference for {model_name} (Sequential Mode + Strong TTA)...")
    
    # --- Sequential Mode: Load -> Predict -> Delete ---
    for fold in range(Config.N_FOLDS):
        path = os.path.join(Config.MODEL_SAVE_DIR, f"{model_name}_fold{fold}_best.pth")
        if not os.path.exists(path):
            print(f"⚠️ Warning: Model {path} not found. Skipping.")
            continue
            
        print(f"Processing Fold {fold}...")
        
        # 1. Load Model
        model = timm.create_model(timm_name, pretrained=False, num_classes=Config.NUM_CLASSES)
        
        # [Fix] Handle torch.compile prefix '_orig_mod.'
        checkpoint = torch.load(path, map_location=device)
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
            
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            # Fallback (Just in case)
            print(f"⚠️ Safe loading failed, trying strict=False. Error: {e}")
            model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        # PyTorch 2.0 Compile (Optional)
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except:
            pass

        # 2. Prediction (with Strong TTA & AMP)
        fold_probs = []
        with torch.no_grad():
            # Enable AMP for faster inference
            with torch.amp.autocast('cuda', enabled=True):
                for images in tqdm(test_loader, desc=f"Fold {fold}", leave=False):
                    images = images.to(device)
                    
                    # Strong TTA Inference
                    probs = tta_inference(model, images)
                    fold_probs.append(probs) # Keep as tensor
        
        # Concatenate this fold's results
        fold_res = torch.cat(fold_probs, dim=0) # (N_test, Num_Classes)
        all_fold_preds.append(fold_res)
        
        # 3. Critical: Delete model and clear VRAM
        del model
        gc.collect()
        torch.cuda.empty_cache() 
        
    if not all_fold_preds:
        return None

    # --- Power Mean Ensemble ---
    print("Applying Power Mean Ensemble (p=1.5)...")
    
    # Stack all folds -> (N_Folds, N_Test, Num_Classes)
    stacked_preds = torch.stack(all_fold_preds)
    
    # Power Mean: (Sum(x^p) / N)^(1/p) ... roughly speaking, or just mean of powers
    # The user suggested: (stacked ** p).mean(0)
    # Technically power mean is different, but in Kaggle context, "Power Ensemble" usually means this:
    # 1. Raise probs to power p
    # 2. Average them
    # 3. Re-normalize (optional but good)
    
    p = 1.5
    ensemble_preds = (stacked_preds ** p).mean(dim=0)
    
    # Normalize rows to sum to 1
    ensemble_preds = ensemble_preds / ensemble_preds.sum(dim=1, keepdim=True)
    
    return ensemble_preds.cpu().numpy()

def generate_pseudo_labels(debug=False):
    test_df = pd.read_csv(Config.TEST_CSV)
    if debug:
        test_df = test_df.sample(100).reset_index(drop=True)
    
    ensemble_probs = None
    model_count = 0
    
    # Run inference for each architecture
    for model_name in Config.MODELS.keys():
        probs = inference_per_model(model_name, test_df, Config.DEVICE)
        
        if probs is not None:
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs
            model_count += 1
            
    if ensemble_probs is None:
        print("No models found!")
        return

    # Average over all architectures
    ensemble_probs /= model_count
    
    # Generate Pseudo Labels
    max_probs = np.max(ensemble_probs, axis=1)
    preds = np.argmax(ensemble_probs, axis=1)
    
    high_confidence_idx = np.where(max_probs > Config.CONFIDENCE_THRESHOLD)[0]
    
    print(f"Total Test Samples: {len(test_df)}")
    print(f"High Confidence Samples (> {Config.CONFIDENCE_THRESHOLD}): {len(high_confidence_idx)}")
    
    # Create Pseudo DataFrame
    pseudo_df = test_df.iloc[high_confidence_idx].copy()
    pseudo_df['Label'] = preds[high_confidence_idx]
    
    # Initial Submission (Optional)
    submission_df = test_df.copy()
    submission_df['Label'] = preds
    submission_df.to_csv(os.path.join(Config.SUBMISSION_DIR, 'submission_step1.csv'), index=False)
    
    # Merge with Train
    train_df = pd.read_csv(Config.TRAIN_CSV)
    
    # Realign columns if needed (Train has ID, Label. Test has ID, Label(empty?))
    # Ensure pseudo_df has same columns as train_df
    pseudo_df = pseudo_df[['ID', 'Label']]
    
    new_train_df = pd.concat([train_df, pseudo_df], axis=0).reset_index(drop=True)
    
    output_path = os.path.join(Config.DATA_DIR, 'train_pseudo.csv')
    new_train_df.to_csv(output_path, index=False)
    print(f"Saved extended dataset to {output_path} (Size: {len(new_train_df)})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    generate_pseudo_labels(args.debug)
