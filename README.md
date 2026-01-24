# Data

### file structure

.data/
-----/[Location]/
----------------currentRaw.csv
----------------dayhour/
-----------------------[species].csv
----------------split/
-----------------------xTrain-[species].csv
-----------------------xTest-[species].csv
-----------------------yTrain-[species].csv
-----------------------yTest-[species].csv

### Information

- Species list: pm10, pm25, no, no2, o3, co

# Settup environment

- python -m venv .venv
- pip install -r Requirements.txt

# Steps to train model for specific sensor:

- download the QuantAQ and GAPA data from May 31st - July 16th, 2025
- preprocess.py collocatedQAQ GAPA [species]
- modelBakeoff.py [sensor#]/xTest.csv [sensor#]/yTest.csv [sensor#]/xTrain.csv [sensor#]/yTrain.csv [sensor#]/bestParams.json

# Steps to correct current data:

- preprocessCurrent.py [species]

# Visualization:

- Confirm model bakeoff works by using the bakeoff's best model
- Visualize test: python visualization.py [TODO]
