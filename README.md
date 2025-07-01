# Driver Behavior Classification using SUMO and CARLA Simulations

This project uses synthetic driving data generated via **SUMO** and **CARLA** simulators to classify driver behavior (e.g., *aggressive*, *normal*). Multiple simulation configurations are used to augment the data and train robust machine learning models. The repository includes simulation tools, data processing pipelines, and evaluation notebooks.

---

## 🚦 Project Structure

```
driver-behavior-simulation/
├── data/ # All datasets
│ ├── base/ # Real-world or comparison dataset
│ ├── synthetic/ # Generated via CARLA and SUMO
│ │ ├── sumo/config_a/
│ │ ├── sumo/config_b/
│ │ ├── carla/config_a/
│ │ └── carla/config_b/
│ ├── merged/ # Datasets combined for experiments
│ └── processed/ # Cleaned and feature-engineered data
│
├── notebooks/ # Jupyter notebooks for all major steps
│ ├── 0_generate_sumo_data.ipynb
│ ├── 0_generate_carla_data.ipynb
│ ├── 1_merge_datasets.ipynb
│ ├── 2_feature_engineering.ipynb
│ ├── 3_model_training.ipynb
│ ├── 4_evaluation.ipynb
│ └── 5_visualization.ipynb
│
├── src/ # Python modules
│ ├── sim/ # Interfaces for CARLA and SUMO
│ │ ├── sumo_utils.py
│ │ └── carla_utils.py
│ ├── data/ # Data loading and preprocessing
│ ├── features/ # Feature extraction
│ ├── models/ # ML training and evaluation
│ └── utils/ # Config, helpers
│
├── configs/ # Simulation config files
│ ├── sumo_config_a.xml
│ ├── sumo_config_b.xml
│ ├── carla_config_a.yaml
│ └── carla_config_b.yaml
│
├── results/ # Experiment logs, figures, metrics
│ ├── logs/
│ ├── metrics/
│ └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```
# Example structure

```
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

```

## 📊 Objective

Simulate and classify driving behavior using two distinct approaches:

1. **Synthetic Data Generation**
   - Generate trajectories using:
     - CARLA behavior agents with various configurations
     - SUMO vehicle models with varied acceleration, deceleration, and routing
   - Label each dataset with the intended driver behavior (e.g., *aggressive*, *normal*)

2. **Machine Learning**
   - Preprocess raw simulation logs into usable features
   - Merge synthetic datasets with real-world baselines
   - Train and evaluate multiple classifiers:
     - LSTM, CNN, Decision Tree, etc.
   - Visualize results with t-SNE, UMAP, confusion matrices

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/driver-behavior-simulation.git
cd driver-behavior-simulation
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
