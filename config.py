import os
import torch

class Config:
    # Directories
    DATA_DIR = './data'
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_images')
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_images')
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
    
    OUTPUT_DIR = './outputs'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
    SUBMISSION_DIR = os.path.join(OUTPUT_DIR, 'submissions')
    
    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 12 # Increased to speed up RAM caching loop
    
    # Training Hyperparameters
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 20  # Adjust based on convergence
    WARMUP_EPOCHS = 3 # Warmup for stable training
    BATCH_SIZE = 12 # Adjust based on VRAM (Example: 8, 16, 32)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    WEIGHT_DECAY = 1e-2
    ACCUMULATION_STEPS = 1
    
    # Speed Optimizations
    CACHE_IMAGES_TO_RAM = True # Load all images to RAM (~3-5GB) to eliminate Disk I/O

    # Models (The "Godly" Trio)
    MODELS = {
        'convnext': 'convnextv2_large.fcmae_ft_in22k_in1k_384', # 384x384
        'eva02': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', # 448x448
        'swinv2': 'swinv2_base_window12_192.ms_in22k'  # 192x192
    }
    
    # Input Sizes for each model
    IMG_SIZES = {
        'convnext': 384,
        'eva02': 448,
        'swinv2': 192
    }

    # Ensemble Weights (Weighted Voting)
    MODEL_WEIGHTS = {
        'convnext': 0.50,  # Main Model
        'swinv2':   0.30,  # Auxiliary (Transformer)
        'eva02':    0.20   # Auxiliary (Large)
    }

    # Pseudo Labeling
    CONFIDENCE_THRESHOLD = 0.99
    
    # Classes
    CLASSES = [
        'normal', 
        'void', 
        'horizontal_defect', 
        'vertical_defect', 
        'edge_defect', 
        'particle'
    ]
    NUM_CLASSES = 6
    
    # Data Distribution (Analyzed from train.csv)
    # 0: Normal, 1: Void, 2: Horizontal, 3: Vertical, 4: Edge, 5: Particle
    CLASS_COUNTS = {
        0: 674,
        1: 492,
        2: 100,  # Warning: Imbalanced!
        3: 378,
        4: 240,
        5: 644
    }
    
    # Class Weights for Loss Function (Inverse Frequency)
    # Can be used with CrossEntropyLoss(weight=torch.tensor(Config.CLASS_WEIGHTS).to(device))
    # Formula: Total / (NumClasses * Count)
    TOTAL_SAMPLES = 2528
    CLASS_WEIGHTS = [
        2528 / (6 * count) for count in CLASS_COUNTS.values()
    ]

if __name__ == '__main__':
    print(f"Device: {Config.DEVICE}")
    print(f"Models: {list(Config.MODELS.keys())}")
