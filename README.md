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
