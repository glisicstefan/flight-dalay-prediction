import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# RANDOM SPLIT

def create_random_splits(input_file_path, output_dir):
    """
    Create random train/validation/test splits.
    
    Args:
        input_filepath: Path to cleaned data (e.g., 'data/interim/flights_cleaned.csv')
        output_dir: Where to save splits (e.g., 'data/processed/')
    
    Returns:
        None (saves 3 CSV files: train.csv, val.csv, test.csv)
    """

    data = pd.read_csv(input_file_path)

    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    if os.path.exists(output_dir):
        print(f"{output_dir} directory exist.")
    else:
        print(f"Creating {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_data)} flights.    | {len(train_data) / len(data) * 100:.1f}%")
    print(f"Validation: {len(val_data)} flights. | {len(val_data) / len(data) * 100:.1f}%")
    print(f"Test: {len(test_data)} flights.      | {len(test_data) / len(data) * 100:.1f}%")



if __name__ == '__main__':
    input_path = "../../data/interim/flights_cleaned.csv"
    output_path = "../../data/processed/"

    create_random_splits(input_file_path=input_path, output_dir=output_path)
    print("\n✅ Splits created successfully!")