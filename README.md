# TAGI-LSTM-smoothing
This repos contains (1) Analytical tractable Bayesian Long-short term memory neural network (TAGI-LSTM); (2) hybrid model which couples TAGI-LSTM and State-space Models (SSM).
1. TAGI-LSTM: run the "synthetic_LSTM_smoothing.m" file
2. Hybrid model:
- To decompose a time series with linear trend: run the "synthetic_coupling_normal.m" file. The model contains a level, a trend and an TAGI-LSTM component.
- To decompose a time series with a complex non-linear trend: run the two examples "synthetic_coupling_exponential_smoothing.m" and "tourismM30_coupling_exponential_smoothing.m". The model contains a a level, a trend, and exponential smoothing and an TAGI-LSTM component.
