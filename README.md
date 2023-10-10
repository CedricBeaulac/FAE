# FAE
This repo contains data sets and python code for the implementations described in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

### The folder **Datasets** consists of
- The folder **RealApplication** that stores the *El Nino* data set applied in the *Real Application* section of the manuscript, including the actual observations (`ElNino_ERSST.csv`), the observed time stamp (`ElNino_ERSST_tpts.csv`) and the randomly customized labels (`ElNino_ERSST_label.csv`).
- The folder **Simulation** that stores the simulation data sets applied in the *Simulation* section of the manuscript, where `sim_s2_data.txt` contains observations/time/label information scenario 1.1 and `sim_s1_data.txt` contains observations/time/label information for Scenario 1.2, 2.1 & 2.2.

### The folder **Code** consists of 
- `AE_irregular.py`: code for implementing the conventional autoencoders (AE) with irregularly spaced functional data.
- `AE_regular.py`: code for implementing the conventional autoencoders (AE) with regularly spaced functional data.
- `DataGenerator_NN.py`: the data generator for simulation data sets.
- `FAE_irregular.py`: code for implementing the proposed functional autoencoders (FAE) with irregularly spaced functional data.
- `FAE_regular.py`: code for implementing the proposed functional autoencoders (FAE) with regularly spaced functional data.
- `FPCA.py`: code for implementing the functional principal component analysis (FPCA) with regularly spaced functional data.
- `Functions.py`: the self-defined functions used for running the existing and proposed methods implemented.
- `Plotting.py`: code for creating the plots displayed in the manuscript.
- `Read_ElNino_Data.py`: code for importing and pre-processing the El Nino data set in the manuscript.
- `Read_Sim_Data.py`: code for importing and pre-processing the simulation data sets in the manuscript.
