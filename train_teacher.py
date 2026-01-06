import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.cuda.amp as amp # Deprecated
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import timm
from timm.utils import ModelEmaV2 
from timm.scheduler import CosineLRScheduler
import gc

from config import Config
from dataset import AOIDataset, get_transforms
from utils import seed_everything, get_folds, get_score, get_balance_sampler

# Optimization: Enable TF32 for faster matrix multiplication on Ampere GPUs
torch.set_float32_matmul_precision('high')

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, scaler, accum_iter, ema_model=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}")
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss = loss / accum_iter
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_iter == 0 or (step + 1) == len(dataloader):
            # [Gradient Clipping] Prevent Mode Collapse
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema_model is not None:
                ema_model.update(model)
        
        # Calculate Accuracy
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        running_acc += acc
        
        running_loss += loss.item() * accum_iter
        
        pbar.set_postfix(loss=running_loss/(step+1), acc=running_acc/(step+1))
    
    return running_loss / len(dataloader), running_acc / len(dataloader)

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    # model here should be ema_model.module if EMA is used
    
    running_loss = 0.0
    preds = []
    actuals = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Valid Epoch {epoch}")
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        running_loss += loss.item()
        preds.append(outputs.argmax(1).cpu())
        actuals.append(labels.cpu())
        
        pbar.set_postfix(loss=running_loss/(step+1))
    
    preds = torch.cat(preds)
    actuals = torch.cat(actuals)
    acc = get_score(actuals.numpy(), preds.numpy())
    
    return running_loss / len(dataloader), acc

def run_training(model_name, debug=False):
    seed_everything(Config.SEED)
    
    TARGET_BATCH_SIZE = Config.BATCH_SIZE
    ACTUAL_BATCH_SIZE = 12 # Optimized for Speed (Try 12 or 16)
    accum_iter = max(1, TARGET_BATCH_SIZE // ACTUAL_BATCH_SIZE)
    
    print(f"Target Batch: {TARGET_BATCH_SIZE}, Actual Batch: {ACTUAL_BATCH_SIZE}, Accumulation Steps: {accum_iter}")
    
    df = pd.read_csv(Config.TRAIN_CSV)
    if debug:
        df = df.sample(100).reset_index(drop=True)
    
    df = get_folds(df, n_folds=Config.N_FOLDS, seed=Config.SEED)
    
    img_size = Config.IMG_SIZES[model_name]
    timm_name = Config.MODELS[model_name]
    
    print(f"Training {model_name} ({timm_name}) with image size {img_size}")
    
    for fold in range(Config.N_FOLDS):
        print(f"--- Fold {fold} ---")
        
        # [Auto-Resume] 1. Check if Fold is already completed
        best_model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{model_name}_fold{fold}_best.pth")
        log_path = os.path.join(Config.OUTPUT_DIR, f"{model_name}_fold{fold}_log.csv")
        
        if os.path.exists(best_model_path) and os.path.exists(log_path):
            try:
                log_df = pd.read_csv(log_path)
                if len(log_df) >= Config.EPOCHS:
                    print(f"⏩ Fold {fold} completed ({len(log_df)} epochs). Skipping...")
                    continue
            except:
                pass

        # Create history list for logging
        history = []
        history = []
        
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        valid_df = df[df['fold'] == fold].reset_index(drop=True)
        
        # [Class Balancing] Create Balanced Sampler
        train_sampler = get_balance_sampler(train_df)
        
        train_dataset = AOIDataset(
            train_df, 
            transforms=get_transforms(img_size, mode='train'),
            mode='train'
        )
        valid_dataset = AOIDataset(
            valid_df, 
            transforms=get_transforms(img_size, mode='valid'),
            mode='valid'
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=ACTUAL_BATCH_SIZE,
            sampler=train_sampler, # Use Sampler
            shuffle=False,         # Must be False when using Sampler
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True # Speed Optimization
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=ACTUAL_BATCH_SIZE * 2, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True # Speed Optimization
        )
        
        model = timm.create_model(
            timm_name, 
            pretrained=True, 
            num_classes=Config.NUM_CLASSES,
            drop_path_rate=0.2 
        )
        
        # Speed Optimization: Disable Gradient Checkpointing (if VRAM allows)
        # model.set_grad_checkpointing(True) 
        model.to(Config.DEVICE)
        
        # Speed Optimization: PyTorch 2.0 Compile
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✅ PyTorch 2.0 Compile Enabled")
        except Exception as e:
            print(f"⚠️ Compile failed: {e}")
            
        # Initialize EMA
        # Reduced decay from 0.9999 to 0.995 for small dataset (~2k images)
        ema_model = ModelEmaV2(model, decay=0.995, device=None)
        
        scaler = torch.amp.GradScaler('cuda')
        
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        
        # Scheduler with Warmup
        scheduler = CosineLRScheduler(
            optimizer, 
            t_initial=Config.EPOCHS, 
            lr_min=1e-6, 
            warmup_t=Config.WARMUP_EPOCHS, 
            warmup_lr_init=1e-6, 
            cycle_limit=1
        )
        
        best_acc = 0.0
        start_epoch = 0
        
        # [Auto-Resume] 2. Check for 'last' checkpoint to resume training
        last_model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{model_name}_fold{fold}_last.pth")
        if os.path.exists(last_model_path):
            print(f"🔄 Resuming Fold {fold} from checkpoint: {last_model_path}")
            checkpoint = torch.load(last_model_path, map_location=Config.DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'ema_state_dict' in checkpoint and ema_model is not None:
                ema_model.module.load_state_dict(checkpoint['ema_state_dict'])
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            
            # Resume history
            if os.path.exists(log_path):
                history = pd.read_csv(log_path).to_dict('records')
            
            print(f"Resuming from Epoch {start_epoch}, Best Acc: {best_acc:.4f}")
        
        for epoch in range(start_epoch, Config.EPOCHS):
            # Pass ema_model to train loop
            train_loss, train_acc = train_one_epoch(
                model, optimizer, scheduler, train_loader, Config.DEVICE, epoch, 
                scaler, accum_iter, ema_model
            )
            
            # Step scheduler after epoch (timm scheduler usually takes epoch index or epoch + 1)
            # For CosineLRScheduler, step(epoch) where epoch starts at 0 for start, 
            # but usually it's called at the end of epoch. 
            # If we call it at end of epoch 0, we should pass 1.
            scheduler.step(epoch + 1)
            
            # Validate using Original model (Optional, but good for comparison)
            valid_loss, valid_acc = valid_one_epoch(
                model, 
                valid_loader, 
                Config.DEVICE, 
                epoch
            )
            
            # Validate using EMA model
            ema_valid_loss, ema_valid_acc = valid_one_epoch(
                ema_model.module, 
                valid_loader, 
                Config.DEVICE, 
                epoch
            )
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Valid Loss={valid_loss:.4f} Acc={valid_acc:.4f} | EMA Loss={ema_valid_loss:.4f} Acc={ema_valid_acc:.4f}")
            
            # Log metrics
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'ema_valid_loss': ema_valid_loss,
                'ema_valid_acc': ema_valid_acc
            })
            pd.DataFrame(history).to_csv(os.path.join(Config.OUTPUT_DIR, f"{model_name}_fold{fold}_log.csv"), index=False)
            
            if ema_valid_acc > best_acc:
                print(f"Validation Accuracy Improved ({best_acc:.4f} -> {ema_valid_acc:.4f}). Saving EMA model...")
                best_acc = ema_valid_acc
                torch.save(
                    ema_model.module.state_dict(), 
                    os.path.join(Config.MODEL_SAVE_DIR, f"{model_name}_fold{fold}_best.pth")
                )
            
            # [Auto-Resume] 3. Save Last Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.module.state_dict() if ema_model else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(Config.MODEL_SAVE_DIR, f"{model_name}_fold{fold}_last.pth"))
        
        del model, ema_model, optimizer, scheduler, scaler, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['convnext', 'eva02', 'swinv2'], help='Model key from Config')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fewer data)')
    args = parser.parse_args()
    
    run_training(args.model, args.debug)
