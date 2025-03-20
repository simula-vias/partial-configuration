#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from common import load_data


def create_fixed_splits(n_splits=4, random_state=42):
    """
    Create fixed splits for each dataset based on inputs.
    
    Args:
        n_splits (int): Number of splits to create
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing splits for each dataset
    """
    # Get list of dataset directories
    data_dir = "./data"
    dataset_dirs = [
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d)) and d not in ["others"]
    ]
    
    # Initialize results dictionary
    all_splits = {}
    
    # Process each dataset
    for system in dataset_dirs:
        print(f"Processing dataset: {system}")
        
        try:
            # Load data for the current system
            (
                perf_matrix,
                input_features,
                _,
                _,
                _,
                _,
            ) = load_data(system=system, data_dir=data_dir, input_properties_type="tabular")
            
            # Get unique input names
            input_names = perf_matrix["inputname"].unique()
            
            # Create KFold cross-validator
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            # Generate splits
            system_splits = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(input_names)):
                # Get train and test input names
                train_inputs = sorted(input_names[train_idx].tolist())
                test_inputs = sorted(input_names[test_idx].tolist())
                
                # Store split information
                split_info = {
                    "fold": fold_idx,
                    "train_inputs": train_inputs,
                    "test_inputs": test_inputs
                }
                system_splits.append(split_info)
            
            # Add to results
            all_splits[system] = system_splits
            print(f"  Created {n_splits} splits for {len(input_names)} inputs")
            
        except Exception as e:
            print(f"  Error processing {system}: {str(e)}")
    
    return all_splits


def read_splits(system, filepath="./data/splits.json"):
    """
    Reads the splits.json file, extracts data for a specific system, and
    returns a list of (train_inp, test_inp) tuples.

    Args:
        system (str): The system name to extract splits for.
        filepath (str): Path to the splits.json file.

    Returns:
        list: List of (train_inp, test_inp) tuples, or None if the system
              is not found in the splits file.
    """
    with open(filepath, "r") as f:
        all_splits = json.load(f)

    if system not in all_splits:
        raise ValueError(f"No splits for system '{system}'.")
    
    system_splits = all_splits[system]
    return [
        (split["train_inputs"], split["test_inputs"]) for split in system_splits
    ]


def main():
    # Create splits
    splits = create_fixed_splits(n_splits=4, random_state=42)
    
    # Save to JSON file
    output_path = Path("./data/splits.json")
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"Splits saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    for system, system_splits in splits.items():
        num_inputs = len(system_splits[0]["train_inputs"]) + len(system_splits[0]["test_inputs"])
        print(f"{system}: {num_inputs} inputs, {len(system_splits)} splits")


if __name__ == "__main__":
    main()
