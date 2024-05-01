close all
clear all
path  = char([cd ,'/data']);

%% Data
initSeed = 1;       % Fixed seed to obtain reproductible results
rng(initSeed)

y1 = 0;
yn = 2*pi;
st = yn/24;
y  = -round(sin(y1:st:yn),2)';
y(end) = [];
y  = repmat(y,[15,1]); 
y = y + randn(size(y))*0.2;

% trend
trend = [linspace(0,2,length(y))]';

y = y+trend;
% plot(y)

t1 = datenum('01-Jan-2000 00:00:00','dd-mmm-yyyy HH:MM:SS');
tend = t1 + 1/24*(size(y,1)-1);
t = [t1:1/24:tend]';

nb_test = 24;
nb_val = 24;
sql = 12;

ytrain = y(1:end-nb_test-nb_val);
ttrain = t(1:end-nb_test-nb_val);
yval = y(end-nb_test-nb_val-sql+1:end-nb_test);
tval = t(end-nb_test-nb_val-sql+1:end-nb_test);
ytest = y(end-nb_test-sql+1:end);
ttest = t(end-nb_test-sql+1:end);

ttrain = datetime(ttrain,'ConvertFrom','datenum');
ttrain = datestr(ttrain, 'yyyy-mm-dd HH:MM:SS');
tval = datetime(tval,'ConvertFrom','datenum');
tval = datestr(tval, 'yyyy-mm-dd HH:MM:SS');
ttest = datetime(ttest,'ConvertFrom','datenum');
ttest = datestr(ttest, 'yyyy-mm-dd HH:MM:SS');

% figure
% plot(ttrain,ytrain,'r')
% hold on
% xline(ttrain(end))
% plot(tval,yval,'k','LineStyle','--')
% xline(tval(end))
% plot(ttest,ytest,'b')


% train
writematrix(ytrain, char([path, '/x_train_sin_trend_matlab.csv']));
writematrix(ttrain, char([path, '/x_train_sin_trend_matlab_datetime.csv']));
% validation
writematrix(yval, char([path, '/x_val_sin_trend_matlab.csv']));
writematrix(tval, char([path, '/x_val_sin_trend_matlab_datetime.csv']));
% test
writematrix(ytest, char([path, '/x_test_sin_trend_matlab.csv']));
writematrix(ttest, char([path, '/x_test_sin_trend_matlab_datetime.csv']));



 
 
 
 
 
 
 
 
 

