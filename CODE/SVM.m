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

%working_predictors = data(:,1:11);
working_predictors = [volatileAcidity residualSugar freeSulfurDioxide sulphates alcohol chlorides pH];

%Support Vector Machines
model_c_svm = fitcecoc(working_predictors(idx_training,:),quality(idx_training));
model_c_svm_optimize = fitcecoc(working_predictors(idx_training,:),quality(idx_training),'OptimizeHyperparameters','auto');
model_c_svm.ClassNames;
model_c_svm.CodingMatrix;
model_c_svm.BinaryLearners{1};
isLoss = resubLoss(model_c_svm);
%crossval(model_c_svm);


%Apply the optimized model on test data
label_test_svm_c = predict(model_c_svm,working_predictors(idx_test,:));

%find the misclassified labels
numMisClass = sum(sign(abs(label_test_svm_c - quality(idx_test))));


svm_test_quality = zeros(max(quality) - min(quality) + 1,numel(idx_test));
label_test_svm_c_mat = svm_test_quality;
for j = 1:numel(idx_test)
    svm_test_quality(quality(idx_test(j))-2,j) = 1;
    label_test_svm_c_mat(label_test_svm_c(j)-2,j) = 1;
end
[c,cm] = confusion(svm_test_quality,label_test_svm_c_mat)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
confusionMat = cm;
precision = diag(cm)./sum(cm,2);
recall = diag(cm')./sum(cm',1)';
f1Scores = 2*(precision.*recall)./(precision+recall)
meanF1 =  mean(f1Scores)

figure;plotroc(svm_test_quality,label_test_svm_c_mat)
set(gcf,'color','white')
set(gca,'FontSize',18)

figure;plotconfusion(svm_test_quality,label_test_svm_c_mat)
