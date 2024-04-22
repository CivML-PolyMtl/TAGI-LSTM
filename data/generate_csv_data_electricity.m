
%% Data
path  = char([cd ,'/data']);
data  = load(char([path, '/electricity.mat']));
tsIdx = [1:370];
y     = data.elec(:,tsIdx);

%% 1 week: electricity_2014_03_31
% time covariates
time_start = datenum('01-Jan-2012 00:00:00','dd-mmm-yyyy HH:MM:SS');
time_end = time_start + 1/24*(size(y,1)-1);
t = [time_start:1/24:time_end]';
t = t(19513:19848);  
y = y(19513:19848,:);

train_end = 168;
sql = 24;
val_len = 24;

ytrain = y(1:train_end-sql,:);
ttrain = t(1:train_end-sql,:);
yval = y(train_end-sql-val_len+1:train_end,:);
tval = t(train_end-sql-val_len+1:train_end,:);
ytest = y(train_end+1-sql:end,:);
ttest = t(train_end+1-sql:end,:);

ttrain = datetime(ttrain,'ConvertFrom','datenum');
ttrain = datestr(ttrain, 'yyyy-mm-dd HH:MM:SS');
tval = datetime(tval,'ConvertFrom','datenum');
tval = datestr(tval, 'yyyy-mm-dd HH:MM:SS');
ttest = datetime(ttest,'ConvertFrom','datenum');
ttest = datestr(ttest, 'yyyy-mm-dd HH:MM:SS');


% train
writematrix(ytrain, char([path, '/electricity_2014_03_31_train.csv']));
writematrix(ttrain, char([path, '/electricity_2014_03_31_train_datetime.csv']));
% validation
writematrix(yval, char([path, '/electricity_2014_03_31_val.csv']));
writematrix(tval, char([path, '/electricity_2014_03_31_val_datetime.csv']));
% test
writematrix(ytest, char([path, '/electricity_2014_03_31_test.csv']));
writematrix(ttest, char([path, '/electricity_2014_03_31_test_datetime.csv']));



 
 
 
 
 
 
 
 
 

