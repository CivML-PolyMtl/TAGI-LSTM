% Calculate metrics for the tourism (monthly) dataset
clc
close all
clear all
path  = char([cd]);

seasonality = 4;
% load data
data   = load(char([path, '/data/tourism_nanTop.mat']));
nb_test_point = 8;
y = data.quarter_values;
ytrain = y(1:end-nb_test_point,:);     % observation
ytest  = y(end-nb_test_point+1:end,:); % observation

% load result
load(char([path, '/result/tourism/tourism_quarter.mat']));

p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest, ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('tourism quarterly ...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')
















