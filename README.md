# Georgia Bias Correction

## Data

- Data is not stored on this repository

### File Structure

```
.data/
└── [Location]/
    ├── collocation
    |   ├── raw.csv
    |   ├── preprocessed-[species].csv
    |   ├── gapa-[species].csv
    |   └── gapa-preprocessed-[species].csv
    ├── current/
    |   ├── preprocessed-[species].csv
    │   └── raw.csv
    ├── hyperParams/
    │   ├── [species].json
    │   └── [species]-scores.csv
    └── split/
        ├── xTrain-[species].csv
        ├── xTest-[species].csv
        ├── yTrain-[species].csv
        └── yTest-[species].csv
```

### Data cleaning

- Confirm Georgia Air Public Access (GAPA) csv file has the expected format:

```
Date,[species]
```

- This may require manually manipulating the header of the csv file

### Species

`pm10`, `pm25`, `no`, `no2`, `o3`, `co`

## Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
```

On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`. Without it, `import xgboost` fails with
"libxgboost.dylib could not be loaded".

## Steps to Train Model

In Claude Code, `/run-pipeline [Location] [species ...]` runs all of these steps for one location, including picking
the model.

1. Download the QuantAQ and GAPA data from May 31st - July 16th, 2025
2. Run preprocessing:
   ```bash
   python preprocess.py [species]
   ```
3. Run model bakeoff:
   ```bash
   python modelBakeoff.py [species]
   ```
4. Pick the best model from .data/[location]/hyperParams/[species]-scores.csv, copy its algorithm and hyperparameters from [species].json into train_model, and run train.py
   ```bash
   python train.py [species]
   ```

## Steps to Correct Current Data

```bash
python preprocessCurrent.py [species]
```

## Visualization

1. Confirm model bakeoff works by using the bakeoff's best model
2. Visualize test:
   ```bash
   python visualization.py [species] -f train or -f current
   ```
