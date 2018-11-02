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

%AdaBoost
%In this experiment, we choose AdaBoostM2 for ensemble classification
cycle = 500;
treeStump = templateTree('MaxNumSplits',1);
tree_c_AdaBoost = fitcensemble(working_predictors(idx_training,:),quality(idx_training),'Method','AdaBoostM2',...
    'NumLearningCycles',cycle,'Learners',treeStump);
tree_c_totalBoost = fitcensemble(working_predictors(idx_training,:),quality(idx_training),'Method','TotalBoost',...
    'NumLearningCycles',cycle,'Learners',treeStump);
tree_c_LPBoost = fitcensemble(working_predictors(idx_training,:),quality(idx_training),'Method','LPBoost',...
    'NumLearningCycles',cycle,'Learners',treeStump);


%Apply the optimized model on test data
label_test_tree_c_AdaBoost = predict(tree_c_AdaBoost,working_predictors(idx_test,:));
label_test_tree_c_totalBoost = predict(tree_c_totalBoost,working_predictors(idx_test,:));
label_test_tree_c_LPBoost = predict(tree_c_LPBoost,working_predictors(idx_test,:));

%find the misclassified labels
numMisClass_ensemble_ada = sum(sign(abs(label_test_tree_c_AdaBoost - quality(idx_test))));
numMisClass_ensemble_total = sum(sign(abs(label_test_tree_c_totalBoost - quality(idx_test))));
numMisClass_ensemble_LP = sum(sign(abs(label_test_tree_c_LPBoost - quality(idx_test))));

figure
plot(resubLoss(tree_c_AdaBoost,'Mode','Cumulative'));
hold on
plot(resubLoss(tree_c_totalBoost,'Mode','Cumulative'),'r');
plot(resubLoss(tree_c_LPBoost,'Mode','Cumulative'),'g');
hold off
xlabel('Number of stumps');
ylabel('Training error');
legend('AdaBoost','TotalBoost','LPBoost','Location','NE');
set(gcf,'color','white')
set(gca,'FontSize',16)

%cross-validation of the ensembles
cvlp = crossval(tree_c_LPBoost,'KFold',5);
cvtotal = crossval(tree_c_totalBoost,'KFold',5);
cvada = crossval(tree_c_AdaBoost,'KFold',5);

figure
plot(kfoldLoss(cvada,'Mode','Cumulative'));
hold on
plot(kfoldLoss(cvtotal,'Mode','Cumulative'),'r');
plot(kfoldLoss(cvlp,'Mode','Cumulative'),'g');
hold off
xlabel('Ensemble size');
ylabel('Cross-validated error');
legend('AdaBoost','TotalBoost','LPBoost','Location','NE');

boost_test_quality = zeros(max(quality) - min(quality) + 1,numel(idx_test));
label_test_boost_c_mat = boost_test_quality;
for j = 1:numel(idx_test)
    boost_test_quality(quality(idx_test(j))-2,j) = 1;
    label_test_boost_c_mat(label_test_tree_c_AdaBoost(j)-2,j) = 1;
end
[c,cm] = confusion(boost_test_quality,label_test_boost_c_mat)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

figure;plotroc(boost_test_quality,label_test_boost_c_mat)
set(gcf,'color','white')
set(gca,'FontSize',16)

figure;plotconfusion(boost_test_quality,label_test_boost_c_mat)
