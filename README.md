# simulated_driving_behavior
Evaluating quality of data augmentation using CARLA and SUMO driving simulators for classifying normal and aggressive behaviors of UAH-driveset.

# Repository Architecture

driver-behavior-simulation/
├── data/
│   ├── base/                        # Real-world or reference dataset
│   ├── synthetic/
│   │   ├── sumo/
│   │   │   ├── config_a/           # SUMO dataset with config A
│   │   │   └── config_b/           # SUMO dataset with config B
│   │   ├── carla/
│   │   │   ├── config_a/           # CARLA dataset with config A
│   │   │   └── config_b/           # CARLA dataset with config B
│   ├── merged/                     # Combined datasets (for experiments)
│   └── processed/                  # Cleaned, standardized datasets
│
├── notebooks/
│   ├── 0_generate_sumo_data.ipynb  # SUMO synthetic generation
│   ├── 0_generate_carla_data.ipynb # CARLA synthetic generation
│   ├── 1_merge_datasets.ipynb      # Merge/aggregate datasets
│   ├── 2_feature_engineering.ipynb # Feature extraction, normalization
│   ├── 3_model_training.ipynb      # Model training & tuning
│   ├── 4_evaluation.ipynb          # Model evaluation, confusion matrix
│   └── 5_visualization.ipynb       # TSNE/UMAP plots, etc.
│
├── src/
│   ├── sim/
│   │   ├── sumo_utils.py           # Helpers for SUMO runs and logging
│   │   └── carla_utils.py          # Helpers for CARLA simulation
│   ├── data/
│   │   ├── loader.py               # Load raw/synthetic/merged datasets
│   │   └── preprocessor.py         # Scaling, filtering, etc.
│   ├── features/
│   │   └── extract_features.py     # Trajectory features, signal processing
│   ├── models/
│   │   ├── train.py                # Train a classifier
│   │   ├── evaluate.py             # Accuracy, F1, etc.
│   │   └── baselines.py            # Rule-based or simpler models
│   └── utils/
│       └── config.py               # Global config and constants
│
├── configs/
│   ├── sumo_config_a.xml
│   ├── sumo_config_b.xml
│   ├── carla_config_a.yaml
│   └── carla_config_b.yaml
│
├── results/
│   ├── logs/
│   ├── metrics/
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore

# 🔑 Design Notes

`data/synthetic/` holds raw generated data from SUMO/CARLA (split by config)

`data/merged/` holds all combined scenarios for ML input

`src/sim/` includes reusable code to launch or parse SUMO/CARLA logs (e.g., TraCI interface, CARLA sensors)

`notebooks/` follow a clean step-by-step data science workflow

`configs/` stores simulation definitions for reproducibility

`results/` is where plots, metrics, logs live

`src/models/` separates training, evaluation, and baseline logic
