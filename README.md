# HBV Model is a bucket type conceptual model 
_Add more info on origin_

Original model is from the course ENVM1502 - river basin Hydrology (Markus Hrachowitz). 

NOTE: the implementation of the memory vector is likely wrong here. The DA technique compensates for this and causes it to work.
For a working version with a proper memory vector, refer to recent developments of the HBV-bmi for [eWaterCycle](https://github.com/Daafip/HBV-bmi)

## Overview of code: 

- `Forcing.txt` contains the real observational data
- `Callibration model.ipynb` contains the code running the function from top to bottom
- `HBVMod.py` contains the actual model
- `MC2_NSE.txt`,`MC2_NSE_log.txt`,`MC2_NSE_sqrt.txt` are three observational runs - MCS of 5000 runs takes a minute thus we store the result to improve experience
- `Weigfun.py` contains a weighting function to add lag to the model
- `Forward model.ipynb` old version of forward model - timestep based - running but not OOP
- `Forward model BMI.ipynb` Newer version: Object Oriented Programming, where the model state is a class instance. running model.update() advances the model timestep according to the Basic Model Interface (BMI).