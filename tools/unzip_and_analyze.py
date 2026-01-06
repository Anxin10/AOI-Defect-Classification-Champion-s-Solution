import zipfile
import os
import pandas as pd
import shutil

def unzip_and_analyze():
    # Updated paths for new file structure
    # Assuming script is run from project root, or we handle relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    zip_path = os.path.join(data_dir, 'aoi.zip')
    extract_to = data_dir
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return

    print(f"Inspecting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Check structure
        file_list = zip_ref.namelist()
        print(f"Top 10 files in zip: {file_list[:10]}")
        
        # Determine if we need to extract to 'data' or if 'data' is inside zip
        # Looking at user input: "aoi_data.zip 檔案包含： train_images.zip, train.csv..."
        # It seems the zip contains the files directly or in a folder.
        
        # Let's just extract to 'data' directory.
        print(f"Extracting to {extract_to}...")
        zip_ref.extractall(extract_to)
        
    print("Extraction complete.")
    
    # Check for inner zips (as per description: train_images.zip inside)
    # The user said: "aoi_data.zip 檔案包含： train_images.zip..."
    # So we might need to unzip those too.
    
    inner_zips = ['train_images.zip', 'test_images.zip']
    for z in inner_zips:
        z_path = os.path.join(extract_to, z)
        if os.path.exists(z_path):
            print(f"Found inner zip: {z_path}. Extracting...")
            with zipfile.ZipFile(z_path, 'r') as zip_ref:
                # Extract to the same data directory
                # Usually train_images.zip contains a folder 'train_images' or just images
                # We want data/train_images/
                
                # Check list
                inner_list = zip_ref.namelist()
                print(f"  Content sample: {inner_list[:3]}")
                
                # If first item doesn't contain a directory separator, it might be loose files
                # If so, we should extract to data/train_images manually?
                # Usually standard dataset zips are nice. Let's assume standard behavior first.
                zip_ref.extractall(extract_to)
            
            # Optional: remove inner zip to save space
            # os.remove(z_path)
    
    # Analyze train.csv
    train_csv_path = os.path.join(extract_to, 'train.csv')
    if os.path.exists(train_csv_path):
        print("-" * 30)
        print("Analyzing train.csv...")
        df = pd.read_csv(train_csv_path)
        print(f"Total Training Samples: {len(df)}")
        
        print("Class Distribution:")
        print(df['Label'].value_counts().sort_index())
        
        print("Missing values:")
        print(df.isnull().sum())
    else:
        print(f"Warning: {train_csv_path} not found.")

    # Analyze test.csv
    test_csv_path = os.path.join(extract_to, 'test.csv')
    if os.path.exists(test_csv_path):
        print("-" * 30)
        print("Analyzing test.csv...")
        df_test = pd.read_csv(test_csv_path)
        print(f"Total Test Samples: {len(df_test)}")
    else:
        print(f"Warning: {test_csv_path} not found.")

if __name__ == "__main__":
    unzip_and_analyze()
