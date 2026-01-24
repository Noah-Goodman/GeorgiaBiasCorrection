# Georgia Bias Correction

## Data

### File Structure

```
data/
└── [Location]/
    ├── currentRaw.csv
    ├── dayhour/
    │   └── [species].csv
    └── split/
        ├── xTrain-[species].csv
        ├── xTest-[species].csv
        ├── yTrain-[species].csv
        └── yTest-[species].csv
```

### Species

`pm10`, `pm25`, `no`, `no2`, `o3`, `co`

## Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
```

## Steps to Train Model

1. Download the QuantAQ and GAPA data from May 31st - July 16th, 2025
2. Run preprocessing:
   ```bash
   python preprocess.py collocatedQAQ GAPA [species]
   ```
3. Run model bakeoff:
   ```bash
   python modelBakeoff.py [sensor#]/xTest.csv [sensor#]/yTest.csv [sensor#]/xTrain.csv [sensor#]/yTrain.csv [sensor#]/bestParams.json
   ```

## Steps to Correct Current Data

```bash
python preprocessCurrent.py [species]
```

## Visualization

1. Confirm model bakeoff works by using the bakeoff's best model
2. Visualize test:
   ```bash
   python visualization.py [TODO]
   ```
