% Calculate metrics for the electricity dataset
clc
close all
clear all
path  = char([cd]);

seasonality = 24;
%% Train/test split 2014_03_31
% load data
data   = load(char([path, '/data/electricity.mat']));
y      = data.elec(19513:19848,:);
ytrain = y(1:end-168,:);     % observation
ytest  = y(end-168+1:end,:); % observation

% load result
tagi = load(char([path, '/result/electricity/electricity_2014_03_31.mat']));
ytestPd  = tagi.ytestPd;
SytestPd = tagi.SytestPd;
p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('electricity train/test split 2014_03_31 ...........')
disp('matlab...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')


% load result
ytestPd = readtable(char([path, '/result/electricity/electricity_2014_03_31_ytestPd_pyTAGI.csv']));
ytestPd = table2array(ytestPd);
SytestPd = readtable(char([path, '/result/electricity/electricity_2014_03_31_SytestPd_pyTAGI.csv']));
SytestPd = table2array(SytestPd);

% ytestPd1 = readtable(char([path, '/result/electricity/electricity_2014_03_31_ytestPd_pyTAGI.csv']));
% ytestPd1 = table2array(ytestPd1);
% SytestPd1 = readtable(char([path, '/result/electricity/electricity_2014_03_31_SytestPd_pyTAGI.csv']));
% SytestPd1 = table2array(SytestPd1);

% ytestPd2 = readtable(char([path, '/result/electricity/electricity_2014_03_31_ytestPd_pyTAGI_1.csv']));
% ytestPd2 = table2array(ytestPd2);
% SytestPd2 = readtable(char([path, '/result/electricity/electricity_2014_03_31_SytestPd_pyTAGI_1.csv']));
% SytestPd2 = table2array(SytestPd2);
% 
% ytestPd = (ytestPd1 + ytestPd2)/2;
% SytestPd = (SytestPd1 + SytestPd2)/2;

p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('pytagi...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')

% %% Train/test split 2014_09_01
% data   = load(char([path, '/data/electricity.mat']));
% y      = data.elec(17545:23544,:);
% ytrain = y(1:end-168,:);      % observation
% ytest  = y(end-168+1:end,:);  % observation
% 
% % load result
% tagi = load(char([path, '/result/electricity/electricity_2014_09_01.mat']));
% ytestPd  = tagi.ytestPd;
% SytestPd = tagi.SytestPd;
% p50_tagi = mt.computeND(ytest,ytestPd);
% p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
% RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
% MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);
% 
% disp('electricity train/test split 2014_09_01 ...........')
% disp(['ND/p50:    ' num2str(p50_tagi)])
% disp(['p90:    ' num2str(p90_tagi)])
% disp(['RMSE:    ' num2str(RMSE_tagi)])
% disp(['MASE:    ' num2str(MASE_tagi)])
% disp('............................................')
% 
% %% Train/test split last_7_days
% data   = load(char([path, '/data/electricity.mat']));
% y      = data.elec;
% ytrain = y(1:end-168,:);     % observation
% ytest  = y(end-168+1:end,:); % observation
% 
% tagi = load(char([path, '/result/electricity/electricity_last_7_days.mat']));
% ytestPd  = tagi.ytestPd;
% SytestPd = tagi.SytestPd;
% p50_tagi = mt.computeND(ytest,ytestPd);
% p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
% RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
% MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);
% 
% disp('electricity train/test split last 7 days ...........')
% disp(['ND/p50:    ' num2str(p50_tagi)])
% disp(['p90:    ' num2str(p90_tagi)])
% disp(['RMSE:    ' num2str(RMSE_tagi)])
% disp(['MASE:    ' num2str(MASE_tagi)])















