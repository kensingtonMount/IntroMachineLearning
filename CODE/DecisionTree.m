%CS 4641 Homework 1
%by Shuai Nie (GTID: 903147679)
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

working_predictors = [volatileAcidity residualSugar freeSulfurDioxide sulphates alcohol chlorides pH];
%%
%Classification Tree

%first grow a tree without any cross-validation
tree_c = fitctree(working_predictors(idx_training,:),quality(idx_training));


tree_c_crossVal = fitctree(working_predictors(idx_training,:),quality(idx_training),'CrossVal','on');
numBranches = @(x)sum(x.IsBranch);
numSplits_all = cellfun(numBranches, tree_c_crossVal.Trained);

figure;
histogram(numSplits_all); %find the average number of splits
figure;view(tree_c_crossVal.Trained{1},'Mode','graph'); %Plot one of the trees from the splits

%Observe the cross-validation error with different leaf sizes
rng('default')
leafs = logspace(1,2,10);
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitctree(working_predictors(idx_training,:),quality(idx_training),'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end
figure;
plot(leafs,err);
set(gcf,'color','white')
set(gca,'FontSize',18)
grid on
xlabel('Min Leaf Size');
ylabel('cross-validated error');

%Test with another number of splits by setting the maximum number of splits
splits =  logspace(1,3,10);
M = numel(splits);
error = zeros(M,1);
for m = 1:M
    ts = fitctree(working_predictors(idx_training,:),quality(idx_training),'MaxNumSplits',splits(m),'CrossVal','on');
    %view(numSplits_half.Trained{1},'Mode','graph');
    error(m) = kfoldLoss(ts);
end
figure;
plot(splits,error);
set(gcf,'color','white')
set(gca,'FontSize',18)
xlabel('Max Split Size');
ylabel('cross-validated error');
grid on


%Calculate the cross-validation error for both models
%classErrorDefault = kfoldLoss(tree_c_crossVal);
%classErrorHalf = kfoldLoss(numSplits_half);

%Observe the cross-validation error with different parent sizes
parent =  logspace(1,2,10);
M = numel(parent);
error = zeros(M,1);
for m = 1:M
    ts = fitctree(working_predictors(idx_training,:),quality(idx_training),'MinParentSize',parent(m),'CrossVal','on');
    %view(numSplits_half.Trained{1},'Mode','graph');
    error(m) = kfoldLoss(ts);
end
figure;
plot(parent,error);
set(gcf,'color','white')
set(gca,'FontSize',18)
xlabel('Min Parent Size');
ylabel('cross-validated error');
grid on

%We need to optimize the classfication tree
tree_c_hyperParam = fitctree(working_predictors(idx_training,:),quality(idx_training),'OptimizeHyperparameters','auto');

%Apply the optimized model on test data
label_test_tree_c = predict(tree_c_hyperParam,data(idx_test,1:11));

%find the misclassified labels
numMisClass = sum(sign(abs(label_test_tree_c - quality(idx_test))));


tree_test_quality = zeros(max(quality) - min(quality) + 1,numel(idx_test));
label_test_tree_c_mat = tree_test_quality;
for j = 1:numel(idx_test)
    tree_test_quality(quality(idx_test(j))-2,j) = 1;
    label_test_tree_c_mat(label_test_tree_c(j)-2,j) = 1;
end
[c,cm] = confusion(tree_test_quality,label_test_tree_c_mat)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
confusionMat = cm;
precision = diag(cm)./sum(cm,2);
recall = diag(cm')./sum(cm',1)';
f1Scores = 2*(precision.*recall)./(precision+recall)
meanF1 =  mean(f1Scores)

figure;plotroc(tree_test_quality,label_test_tree_c_mat)
set(gcf,'color','white')
set(gca,'FontSize',18)
