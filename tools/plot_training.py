import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(model_name='convnext'):
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    
    # Find log files
    log_files = sorted(glob.glob(os.path.join(output_dir, f"{model_name}_fold*_log.csv")))
    
    if not log_files:
        print(f"No log files found for {model_name} in {output_dir}")
        return

    print(f"Found {len(log_files)} logs for {model_name}")

    # Initialize plots
    plt.style.use('bmh') # Clean style
    
    # Figure 1: Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    
    # Figure 2: Loss
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(log_files))
    
    best_results = []

    for i, log_file in enumerate(log_files):
        fold = i 
        df = pd.read_csv(log_file)
        
        # Plot Accuracy
        ax_acc.plot(df['epoch'], df['valid_acc'], label=f'Fold {fold} Valid', linestyle='--', color=colors[i], alpha=0.7)
        ax_acc.plot(df['epoch'], df['ema_valid_acc'], label=f'Fold {fold} EMA', linestyle='-', color=colors[i], linewidth=2)
        
        # Plot Loss
        ax_loss.plot(df['epoch'], df['train_loss'], label=f'Fold {fold} Train', linestyle=':', color=colors[i], alpha=0.7)
        ax_loss.plot(df['epoch'], df['valid_loss'], label=f'Fold {fold} Valid', linestyle='-', color=colors[i])
        
        # Record best
        best_ema = df['ema_valid_acc'].max()
        best_epoch = df['ema_valid_acc'].idxmax()
        best_results.append({'fold': fold, 'best_acc': best_ema, 'epoch': best_epoch})

    # Finalize Accuracy Plot
    ax_acc.set_title(f'{model_name} - Validation Accuracy per Fold')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_acc.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f'{model_name}_accuracy_chart.png')
    fig_acc.savefig(acc_path, bbox_inches='tight')
    plt.close(fig_acc)
    
    # Finalize Loss Plot
    ax_loss.set_title(f'{model_name} - Train vs Valid Loss per Fold')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_loss.set_ylim(0, 2.0) # Zoom in
    ax_loss.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f'{model_name}_loss_chart.png')
    fig_loss.savefig(loss_path, bbox_inches='tight')
    plt.close(fig_loss)
    
    print(f"Charts saved to:\n  - {acc_path}\n  - {loss_path}")
    
    # Print Summary Table
    print("\n--- Training Summary ---")
    summary_df = pd.DataFrame(best_results)
    print(summary_df.to_string(index=False))
    print(f"Average Best Acc: {summary_df['best_acc'].mean():.5f}")

if __name__ == "__main__":
    plot_training_results()
