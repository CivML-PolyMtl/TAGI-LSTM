
%% Data
path  = char([cd ,'/data']);
data  = load(char([path, '/traffic.mat']));
tsIdx = [1:963];
y     = data.traffic(:,tsIdx);


%% 1 week: traffic_2008_01_14
% time covariates
time_start = datenum('01-Jan-2008 01:00:00','dd-mmm-yyyy HH:MM:SS');
time_end   = datenum('31-Mar-2009 00:00:00','dd-mmm-yyyy HH:MM:SS');
t = [time_start:1/24:time_end]';
t = t(137:472); 
y = y(137:472,:);

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
writematrix(ytrain, char([path, '/traffic_2008_01_14_train.csv']));
writematrix(ttrain, char([path, '/traffic_2008_01_14_train_datetime.csv']));
% validation
writematrix(yval, char([path, '/traffic_2008_01_14_val.csv']));
writematrix(tval, char([path, '/traffic_2008_01_14_val_datetime.csv']));
% test
writematrix(ytest, char([path, '/traffic_2008_01_14_test.csv']));
writematrix(ttest, char([path, '/traffic_2008_01_14_test_datetime.csv']));



 
 
 
 
 
 
 
 
 

