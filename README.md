# Coupling LSTM Neural Networks and State-Space Models through Analytically Tractable Inference
This repos contains the matlab codes to reproduce the results for the paper: 


*Vuong, Nguyen & Goulet (2024), Coupling LSTM Neural Networks and State-Space Models through Analytically Tractable Inference*, International Journal of Forecasting.

(1) To load the saved predictions and calculate the test metrics:
run scripts in the `/metrics` folder, e.g. `metrics_electricity.m`

(2) To run the code and obtain the predictions for each dataset:
run scripts in the `/config` folder , e.g. `electricity_2014_03_31.m`

(3) To run examples using TAGI-LSTM and the TAGI-LSTM/SSM hybrid model:
runs scripts in the `/examples` folder
- The `synthetic_LSTM_smoothing.m` file is to perform smoothing in TAGI-LSTM. In this example, smoothing is used to infer the past observations before the training time.
- The `synthetic_coupling_normal.m` file is to decompose a time series with linear trend using the TAGI-LSTM/SSM hybrid model. 
-  The `synthetic_coupling_exponential_smoothing.m` and `tourismM30_coupling_exponential_smoothing.m` files are to decompose time series with a complex non-linear trend using the TAGI-LSTM/SSM hybrid model. 

The Python implementation of the TAGI-LSTM method can be found in the pyTAGI library at https://www.tagiml.com/


