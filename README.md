# HBV Model is a bucket type conceptual model 
_Add more info on origin_

Original model is from the course ENVM1502 - river basin Hydrology (Markus Hrachowitz). 

## Overview of code: 

- `Forcing.txt` contains the real observational data
- `Callibration model.ipynb` contains the code running the function
- `HBVMod.py` contains the actual model
- `MC2_NSE.txt`,`MC2_NSE_log.txt`,`MC2_NSE_sqrt.txt` are three observational runs - MCS of 5000 runs takes a minute thus we store the result to improve experience
- `Weigfun.py` contains a weighting function to add lag to the model
- `Forward model.ipynb` current progress on implementing a forward model