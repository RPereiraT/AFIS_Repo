"""
Dataset Configurations for AFIS Experiments

Centralized configuration for all datasets used in regression experiments.
"""

import os

# =============================================================================
# Dataset Directory
# =============================================================================
# Get the directory where this config file lives (experiments/)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(_CONFIG_DIR, 'datasets')


# =============================================================================
# Dataset Configurations
# =============================================================================
# Each dataset configuration contains:
# - file: pickle file name (relative to datasets/ folder)
# - n_fuzzy_part_dim: number of fuzzy partitions per dimension
# - n_rules: target number of rules after filtering
# - skip_rows: rows to skip at the beginning (for time series)
# - description: brief description of the dataset

DATASETS = {
    'abalone': {
        'file': 'abalone_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 39,
        'skip_rows': 0,
        'description': 'Predict age (rings) of abalone from physical measurements'
    },
    'concrete': {
        'file': 'concrete_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 36,
        'skip_rows': 0,
        'description': 'Predict compressive strength (MPa) of concrete'
    },
    'concrete_slump': {
        'file': 'concrete_slump_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 25,
        'skip_rows': 0,
        'description': 'Predict slump (cm) of concrete'
    },
    'mackey_glass': {
        'file': 'mg_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 40,
        'skip_rows': 20,
        'description': 'Chaotic time series prediction (Mackey-Glass equation)'
    },
    'chemical_conc': {
        'file': 'chemical_conc_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 21,
        'skip_rows': 0,
        'description': 'Predict chemical concentration'
    },
    'chemical_temp': {
        'file': 'chemical_temp_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 18,
        'skip_rows': 0,
        'description': 'Predict chemical process temperature'
    },
    'gas_furnace': {
        'file': 'gas_furnace_df.pkl',
        'n_fuzzy_part_dim': 6,
        'n_rules': 66,
        'skip_rows': 0,
        'description': 'Predict CO2 concentration in gas furnace'
    },
    'laser': {
        'file': 'laser_df.pkl',
        'n_fuzzy_part_dim': 7,
        'n_rules': 120,
        'skip_rows': 0,
        'description': 'Chaotic laser time series prediction'
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_dataset_path(dataset_name):
    """
    Get the full path to a dataset file.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (key in DATASETS dict).
        
    Returns
    -------
    path : str
        Full path to the pickle file.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASETS.keys())}")
    return os.path.join(DATASETS_DIR, DATASETS[dataset_name]['file'])


def get_dataset_config(dataset_name):
    """
    Get the configuration for a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
        
    Returns
    -------
    config : dict
        Dataset configuration dictionary.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASETS.keys())}")
    return DATASETS[dataset_name].copy()


def list_datasets():
    """Print available datasets with descriptions."""
    print("Available Datasets:")
    print("-" * 60)
    for name, config in DATASETS.items():
        desc = config.get('description', 'No description')
        print(f"  {name:15s} - {desc}")
    print("-" * 60)

