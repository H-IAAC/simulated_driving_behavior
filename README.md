# Driver Behavior Classification and Synthetic Data Quality Evaluation using SUMO and CARLA Simulators

This project uses synthetic driving data generated via **SUMO** and **CARLA** simulators to augment **UAH-DriveSet** and classify driver behavior (e.g., *aggressive*, *normal*). Multiple simulation configurations are used to augment the data and train robust machine learning models. The repository includes simulation tools, data processing pipelines, and evaluation notebooks.

This project was developed as part of the Cognitive Architectures research line from 
the Hub for Artificial Intelligence and Cognitive Architectures (H.IAAC) of the State University of Campinas (UNICAMP).
See more projects from the group [here](https://github.com/brgsil/RepoOrganizer).

[![](https://img.shields.io/badge/-H.IAAC-eb901a?style=for-the-badge&labelColor=black)](https://hiaac.unicamp.br/)
[![](https://img.shields.io/badge/-Arq.Cog-black?style=for-the-badge&labelColor=white&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4gPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1Ni4wMDQiIGhlaWdodD0iNTYiIHZpZXdCb3g9IjAgMCA1Ni4wMDQgNTYiPjxwYXRoIGlkPSJhcnFjb2ctMiIgZD0iTTk1NS43NzQsMjc0LjJhNi41Nyw2LjU3LDAsMCwxLTYuNTItNmwtLjA5MS0xLjE0NS04LjEtMi41LS42ODksMS4xMjNhNi41NCw2LjU0LDAsMCwxLTExLjEzNi4wMjEsNi41Niw2LjU2LDAsMCwxLDEuMzY4LTguNDQxbC44LS42NjUtMi4xNS05LjQ5MS0xLjIxNy0uMTJhNi42NTUsNi42NTUsMCwwLDEtMi41OS0uODIyLDYuNTI4LDYuNTI4LDAsMCwxLTIuNDQzLTguOSw2LjU1Niw2LjU1NiwwLDAsMSw1LjctMy4zLDYuNDU2LDYuNDU2LDAsMCwxLDIuNDU4LjQ4M2wxLC40MSw2Ljg2Ny02LjM2Ni0uNDg4LTEuMTA3YTYuNTMsNi41MywwLDAsMSw1Ljk3OC05LjE3Niw2LjU3NSw2LjU3NSwwLDAsMSw2LjUxOCw2LjAxNmwuMDkyLDEuMTQ1LDguMDg3LDIuNS42ODktMS4xMjJhNi41MzUsNi41MzUsMCwxLDEsOS4yODksOC43ODZsLS45NDcuNjUyLDIuMDk1LDkuMjE4LDEuMzQzLjAxM2E2LjUwNyw2LjUwNywwLDAsMSw1LjYwOSw5LjcyMSw2LjU2MSw2LjU2MSwwLDAsMS01LjcsMy4zMWgwYTYuNCw2LjQsMCwwLDEtMi45ODctLjczMmwtMS4wNjEtLjU1LTYuNjgsNi4xOTIuNjM0LDEuMTU5YTYuNTM1LDYuNTM1LDAsMCwxLTUuNzI1LDkuNjkxWm0wLTExLjQ2MWE0Ljk1LDQuOTUsMCwxLDAsNC45NTIsNC45NUE0Ljk1Nyw0Ljk1NywwLDAsMCw5NTUuNzc0LDI2Mi43MzlaTTkzNC44LDI1Ny4zMjVhNC45NTIsNC45NTIsMCwxLDAsNC4yMjEsMi4zNDVBNC45Myw0LjkzLDAsMCwwLDkzNC44LDI1Ny4zMjVabS0uMDIyLTEuNThhNi41MTQsNi41MTQsMCwwLDEsNi41NDksNi4xTDk0MS40LDI2M2w4LjA2MSwyLjUuNjg0LTEuMTQ1YTYuNTkxLDYuNTkxLDAsMCwxLDUuNjI0LTMuMjA2LDYuNDQ4LDYuNDQ4LDAsMCwxLDIuODQ0LjY1bDEuMDQ5LjUxOSw2LjczNC02LjI1MS0uNTkzLTEuMTQ1YTYuNTI1LDYuNTI1LDAsMCwxLC4xMTUtNi4yMjksNi42MTgsNi42MTgsMCwwLDEsMS45NjYtMi4xMzRsLjk0NC0uNjUyLTIuMDkzLTkuMjIyLTEuMzM2LS4wMThhNi41MjEsNi41MjEsMCwwLDEtNi40MjktNi4xbC0uMDc3LTEuMTY1LTguMDc0LTIuNS0uNjg0LDEuMTQ4YTYuNTM0LDYuNTM0LDAsMCwxLTguOTY2LDIuMjY0bC0xLjA5MS0uNjUyLTYuNjE3LDYuMTMxLjc1MSwxLjE5MmE2LjUxOCw2LjUxOCwwLDAsMS0yLjMsOS4xNjRsLTEuMS42MTksMi4wNiw5LjA4NywxLjQ1MS0uMUM5MzQuNDc1LDI1NS43NSw5MzQuNjI2LDI1NS43NDQsOTM0Ljc3OSwyNTUuNzQ0Wm0zNi44NDQtOC43NjJhNC45NzcsNC45NzcsMCwwLDAtNC4zMTYsMi41LDQuODg5LDQuODg5LDAsMCwwLS40NjQsMy43NjIsNC45NDgsNC45NDgsMCwxLDAsNC43NzktNi4yNjZaTTkyOC43LDIzNS41MzNhNC45NzksNC45NzksMCwwLDAtNC4zMTcsMi41LDQuOTQ4LDQuOTQ4LDAsMCwwLDQuMjkxLDcuMzkxLDQuOTc1LDQuOTc1LDAsMCwwLDQuMzE2LTIuNSw0Ljg4Miw0Ljg4MiwwLDAsMCwuNDY0LTMuNzYxLDQuOTQsNC45NCwwLDAsMC00Ljc1NC0zLjYzWm0zNi43NzYtMTAuMzQ2YTQuOTUsNC45NSwwLDEsMCw0LjIyMiwyLjM0NUE0LjkyMyw0LjkyMywwLDAsMCw5NjUuNDc5LDIyNS4xODdabS0yMC45NTItNS40MTVhNC45NTEsNC45NTEsMCwxLDAsNC45NTEsNC45NTFBNC45NTcsNC45NTcsMCwwLDAsOTQ0LjUyNywyMTkuNzcyWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTkyMi4xNDMgLTIxOC4yKSIgZmlsbD0iIzgzMDNmZiI+PC9wYXRoPjwvc3ZnPiA=)](https://github.com/brgsil/RepoOrganizer)

## Objective

Provide tools to generate synthetic data from SUMO and CARLA with easily adjustable parameters for Driver Behavior Classification, as well as evaluation methods to analyse the quality of the data generated. 

1. **Synthetic Data Generation**
   - Generate trajectories using:
     - CARLA Traffic Manager with various configurations.
     - SUMO vehicles with various configurations.

2. **Data Evaluation**
   - Merge synthetic datasets with real-world dataset. In this case, UAH-DriveSet.
   - _Train on Real, Test on Synth (TRTS)_: Uses UAH-DrivesSet data for training and synthetic data for validation. We trained RF, SVC and XGB models with hyperparameter sweep. Good performance on this test indicate realism and utility of the data, since the knowledge obtained from real data is applicable to the synthetic data.
    - _Discriminative Score (DS)_: Uses a merge of real and synthetic data, labeled as real or synthetic, for training and validation. Only the RF model was used in this test, since it got an ideal performance, but it should be easy to use any other models. A good performance in this task indicates that the real data is easily distinguishable from the synthetic data, measuring its realism.
    - _Predictive Score (PS)_: Uses real data augmented with synthetic data for training and real data for validation. RF, SVC, and XGB models were trained, using hyperparameter sweep, for 20\%, 60\% and 100\% synthetic data augmentations. In case the models trained with augmented data show a better performance than those trained on real data only, the synthetic data have utility and coherence.
    _t-SNE Visualization_: t-SNE projection allows us to verify the realism and diversity of the data. If there is a large intersection between real and synthetic data in the projection, we are able to suppose the synthetic data has good realism and diversity.
---

## Repository Structure
```
driver-behavior-simulation/
├── data/ # All datasets
│ ├── base/ # UAH-Driveset
│ ├── synthetic/Town01 # Generated via CARLA and SUMO
│ │ ├── sumo # SUMO simulation data and metadata
│ │ └── carla # CARLA simulation data and metadata
│ └── merged/ # Datasets combined for experiments
│
├── notebooks/ # Jupyter notebooks for all major steps
│ ├── 0_generate_carla_files.ipynb # Generating files for CARLA simulation and routines
│ ├── 0_generate_carla_data.ipynb # Running CARLA simulation
│ ├── 1_generate_sumo_files.ipynb # Generating files for SUMO simulation
│ ├── 1_generate_sumo_data.ipynb # Running SUMO simulation
│ ├── 2_merge_datasets.ipynb # Merging synthetic and real datasets
│ ├── 3_model_training_mlflow.ipynb # Training models with MLFlow
│ ├── 3_model_training.ipynb # Training best models for evalution
│ └── 4_evaluation.ipynb # Models evaluation
│
├── src/ # Python modules
│ ├── sim/ # Interfaces for CARLA and SUMO
│ │ ├── carlaDriverBehParameters.csv # All possible CARLA parameters
│ │ ├── sumoDriverBehParameters.csv # All possible SUMO parameters
│ │ ├── llm_routines # Routines generated by 0_generate_carla_files.ipynb
│ │ ├── sumo_utils.py
│ │ ├── sumo_helper.py
│ │ ├── sumo_simulation.py
│ │ ├── llama_connect.py
│ │ ├── sumo_utils.py
│ │ └── carla_utils.py
│ ├── data/ # Data loading and preprocessing
│ │ ├── loader.py
│ │ └── preprocessor.py
│ └── sumo_map/Town01 # SUMO map and simulation files 
│
├── configs/ # Simulation config files
│ ├── carla_fixed.json
│ ├── carla_llm.json
│ ├── sumo_fixed.json
│ └── sumo_llm.json
│
├── results/ # Experiment logs, figures, metrics
│ ├── metrics/ # Dataframe for TRTS, DS and PS scores
│ └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dependencies / Requirements

### Simulators Set-Up

**Groq Key**: We are using Groq to connect to the `gemma2-9b-it model`, as shown below, meaning you must set an environment variable called `"GROQ_API_KEY"` to be able to make LLM requests. The key is free and you can get one at [this link](https://console.groq.com/keys).

```
client = Groq(
    # Initialize the Groq client with the API key from environment variables
    # Ensure that the environment variable GROQ_API_KEY is set with your Groq API key
    # You can set this in your terminal or in a .env file
    api_key=os.getenv("GROQ_API_KEY"),
)
```
**CARLA Simulator Install**: CARLA simulator must be downloaded and installed. This can be done following [this tutorial](https://carla.readthedocs.io/en/latest/start_quickstart/).

**SUMO Simulator Install**: SUMO simulator must also be installed. Follow [this tutorial](https://sumo.dlr.de/docs/Installing/index.html).

### UAH-Driveset Data

We do not have the license to provide UAH-DriveSet in the repository. It must be downloaded from [this link](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/) and placed under the `data` folder interely. After this, you should have `data/base/UAH-DRIVESET-v1`.

## Installation / Usage

First, clone the repo and install the requirenments (either recreating the conda environments or installing the requirements) as follows:
```bash
git clone https://github.com/RenanFlorencio/simulated_driving_behavior
cd simulated_driving_behavior
cd .envs
# Recreate the conda environment (recommended)
conda env create -f carla-env.yml # Replace environment.yml by sumo-env.yml, carla-env.yml and base-env.yml
conda env create -f sumo-env.yml
conda env create -f base-env.yml

# Or else install the requirements in your environment (not recommended)
pip install -r carla-env_requirements.txt
pip install -r sumo-env_requirements.txt
pip install -r base-env_requirements.txt
```

This repository is structured in a way that enables the generation of both CARLA and SUMO synthetic data.

All the noteooks can be found at the `notebooks` folder.
In order to run the entire experiment, one must start with the CARLA data generation step, where routines for the map are generated.

If you want to use another map, you must add the SUMO map files (basically the .sumocfg and .net files) under the `src/sumo_map` folder, where the `Town01` and `Town05` folders can be found, and provide the interest points in CARLA (they can be set using the `src/sim/carla_interest_point_setter.ipynb` notebook). Every notebook has a variable `FOLDER_NAME` or `town` in the first cell that is used to indicate what is the map being used. `Town01` and `Town05` files converted from CARLA can be found at the CARLA repository, under the co-simulation examples, and although both are provided in this repository, only `Town01` was used in the experiment.

### Parameters Configuration

All the parameters used in the experiment can be found in the `configs` folder. The `_fixed` sulfix means it has been set manually and you are free to change them; the `_llm` sulfix means it was generated using the LLM.

The LLM in fact gives a probability distribution for each parameter for each behavior. We then sample from the distribution to create as many set of parameters as desired. The distributions can be seen at the `_dists` files, the minimum and max values are used as 5% tais of a normal distribution.

Important: When running a new experience, always double-check the parameters after asking the LLM to generate the parameters, as it will sometimes hallucinate and create absurd values.

### CARLA Data Generation

For the CARLA data generation, one should use the `carla-env`, available at the envs folder.

The fixed parameters can be found adjusted at the `configs` folder.

`0_generate_carla_files.ipynb`: Used to generate routines in the CARLA map, given interest points provided in the `src/sumo_map/Town01/interest_points.csv` file, and LLM parameters for the CARLA drivers, given the `src/sim/carlaDriverBehParameters.csv` file. If you want to change the parameters generated by the LLM, simply edit the `src/sim/carlaDriverBehParameters.csv` file. The routines will be used in the SUMO notebook and are stored under `src/sim/llm_routines`.

`0_generate_carla_data.ipynb`: Used to run the simulations and save the data. It allows for the easy configuration of several simulation parameters, such as frequency and number of backgound vehicles. The data generated will be saved under the `data/synthetic/carla` folder.

### SUMO Data Generation

For the SUMO data generation, one should use the `sumo-env`, available at the envs folder.

`1_generate_sumo_files.ipynb`: Used to generate the SUMO files for the routines and parameters, given the `src/sim/sumoDriverBehParameters.csv` file. It allows for several configurations such as number of generated set of parameters for each driver behavior, number of background vehicles and etc. The script will generate two `vTypeDistribution`, `finaltrips` and `merged` files, one for fixed and the other for LLM parameters. The `finaltrips` files contain only the vehicles of interest, those which will be used for sensor readings, and the `merged` files contain both the vehicles of interest and the background vehicles generated. If you want to change the parameters generated by the LLM, simply edit the `src/sim/sumoDriverBehParameters.csv` file.

`1_generate_sumo_data.ipynb`: Used to run the simulations and save the data. It is pretty simple and enables determining the frequency of collection of the data. Most of the SUMO configurations are done when generating the files.

### Merging Datasets

Here, you may use the `base-env`, available at the envs folder.

The `2_merge_datastes.ipnyb` notebook is used to load the data from the UAH-DriveSet, break it into training and validation sets, and merge it with the synthetic data generated. Here, the aggressive and normal behavior datas generated from the simulators are also merged.

It is important to note that this is where you define which columns, or sensors, are going to be merged together. This is relavant because SUMO and CARLA collect a different set of sensors.

After it has runned, the data will be saved at `data/merged` with four folders: `carla` (only carla data, merged together into fixed and llm parameters), `sumo` (same as before, but for sumo), `carla_uah` (CARLA and UAH merged data), `sumo_uah` (SUMO and UAH merged data).

### Model Training

Here, you may use the `base-env`, available at the envs folder.

The model training is divided into the `3_model_training_mlflow.ipynb` notebook, which was used to run all the parameter sweeps and log the experiments to MLFLow, and the `3_model_training.ipynb` notebook, which is used to train only the best models found through the MLFLow experiments.

For this reason, **the mlflow notebook is completely optional** to reproduce the results of this experiment, even though it will be useful when running different experiments.

The results of the `3_model_training.ipynb` notebook, comprising the results of TRTS, DS and PS tests will be stored as tables under `results/metrics`.

### Evaluation

Here, you may use the `base-env`, available at the envs folder.

This notebook is used solely to show the results of each test in a simple and clean manner. It displays the tables for TRTS, PS and DS, and shows the tSNE plots for each dataset.

## Results

The full results and explanations can be found at the published article related to this experiment, the values obtained from the tests are as follows:

### TRTS results

Results of TRTS for the best models. None of the models trained on real data had good performance on synthetic data, indicating they are not realist and have low utility. The best model was XGB for all the datasets.

<img width="600" height="150" alt="image" src="https://github.com/user-attachments/assets/e5d262ac-1efa-470d-8e75-e7930581af54" />

### Predictive Score (PS) results

Results of PS for the best models. Sulfixes `fixed` and `llm` indicate the source of the parameters. Sulfixed 20, 60 and 100 indicate the percentage of synthetic data when data augmentation was used. The best performance was that of the model which did not receive any synthetic data. Best model: ⋆ RF; † XGB; ◇ SVC

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/e1e54ef6-0aa5-4daf-9e8e-53652e996e96" />

### Discriminative Score (DS) results

Results of DS for the best models. All models are Random Forests and got perfect accuracy, meaning the real and synthetic data are easily distinguishable.

<img width="600" height="150" alt="image" src="https://github.com/user-attachments/assets/b47b60bb-4c78-4938-928a-f61e53a0cd07" />

### tSNE projections

<img width="600" height="500" src=results/figures/tsne/tsne_results.png />

## Citation

<!--Don't remove the following tags, it's used for placing the generated citation from the CFF file-->
<!--CITATION START-->
```bibtext
@software{my_citation,
author = {da Silva Florencio, Renan Matheus and Dornhofer Paro Costa, Paula},
title = {simulated_driving_behavior},
url = {https://github.com/H-IAAC/simulated_driving_behavior}
}
```
<!--CITATION END-->
## Authors
  
- (2024 - today) Renan Matheus da Silva Florencio: Computer Engineering, UNICAMP
- (Advisor, 2024 - today) Paula Dornhofer Paro Costa: Professor, FEEC-UNICAMP
  
## Acknowledgements

Project supported by the brazilian Ministry of Science, Technology and Innovations, with resources from Law No. 8,248, of October 23, 1991.

---





