%Neural Network
clc;clear;

%%
prompt = ['Which dataset do you wish to open?',...
         '\nChoose from the following:',...
         '\n1. White wine;'...
         '\n2. Red wine;'...
         '\nAnd your choice is:'];
str = input(prompt,'s');
switch str
    case '1'
        FileName = 'winequality-white.csv';
    case '2'
        FileName = 'winequality-red.csv';
end

fid = fopen(FileName, 'r');
if fid == -1, error('Cannoten read file: %s', FileName); end
fgetl(fid);  % Skip first line
data = fscanf(fid, '%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f', [12, inf]).';
fclose(fid);

%Normalize the data
data_raw = data;
for l = 1:size(data,2)
    for m = 1:size(data,1)
    data(m,l) = (data(m,l) - mean(data(:,l)))/std(data(:,l));
    end
end

%Categorize all data
fixedAcidity = data(1:end,1);
volatileAcidity = data(1:end,2);
citricAcid = data(1:end,3);
residualSugar = data(1:end,4);
chlorides = data(1:end,5);
freeSulfurDioxide = data(1:end,6);
totalSulfurDioxide = data(1:end,7);
density = data(1:end,8);
pH = data(1:end,9);
sulphates = data(1:end,10);
alcohol = data(1:end,11);
quality = data_raw(1:end,12);

%Randomly choose 80% data as training data
idx_rand = randperm(numel(quality));
idx_training = idx_rand(1:round(numel(quality)*.8));
%Put the rest 20% data as test data
idx_test = idx_rand(round(numel(quality)*.8)+1:end);

working_predictors = data(:,1:11);
%working_predictors = [volatileAcidity residualSugar freeSulfurDioxide sulphates alcohol chlorides pH];

nn_train_input = working_predictors(idx_training,:)';
%We need to convert the output quality data from 3 to 9 to a matrix with 1
%indicating the level they are in
nn_train_output = zeros(max(quality) - min(quality) + 1,numel(idx_training));
for i = 1:numel(idx_training)
    nn_train_output(quality(idx_training(i))-2,i) = 1;
end
%Also convert the quality of test data into a matrix
nn_test_quality = zeros(max(quality) - min(quality) + 1,numel(idx_test));
for j = 1:numel(idx_test)
    nn_test_quality(quality(idx_test(j))-2,j) = 1;
end
%Set the "random" initial weights to avoid getting slightly different
%results every time it runs (It is optional.)
setdemorandstream(391418381)
net = patternnet([60 30 10]);
view(net)
[net,tr] = train(net,nn_train_input,nn_train_output);
nntraintool
plotperform(tr)

%test
test_output = net(working_predictors(idx_test,:)');
testIndices = vec2ind(test_output)

figure;
plotconfusion(nn_test_quality,test_output);

[c,cm] = confusion(nn_test_quality,test_output)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

figure;plotroc(nn_test_quality,test_output);
