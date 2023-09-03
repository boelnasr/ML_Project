close all;
clear all;
clc;

% Load MNIST dataset
data = load('mnist.mat');

% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool;
end
%% Define training and testing sets
training_set = data.training;
test_set = data.test;
trainData = double(reshape(training_set.images, [], size(training_set.images, 3))');
testData = double(reshape(test_set.images, [], size(test_set.images, 3))');
trainLabels = training_set.labels;
testLabels = test_set.labels;

%% Train Random Forest Classifier
tic
numTrees = 200; % Number of trees in the forest
mdl = TreeBagger(numTrees, trainData, trainLabels, ...
    'Method', 'classification','MinLeafSize',1,SplitCriterion='deviance');

% Predict using the trained model
predictions = predict(mdl, testData);
time_forest = toc
% Convert predictions to numeric format
predictions = str2double(predictions);

%% Calculate accuracy
accuracy = sum(predictions == testLabels) / numel(testLabels);
fprintf('Random Forest Accuracy: %.2f%%\n', accuracy * 100);

%% Confusion Matrix
C = confusionmat(testLabels, predictions);
confusionchart(C);
title('Confusion Matrix for Random Forest');