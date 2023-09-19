close all;
clear all;
clc;

% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool;
end
%% Load MNIST dataset
data = load('mnist.mat');

%% Define training and testing sets
training_set = data.training;
test_set = data.test;

%% Reshape images into vectors
trainData = reshape(training_set.images, [], size(training_set.images, 3))';
testData = reshape(test_set.images, [], size(test_set.images, 3))';

%% Convert labels to categorical
trainLabels = categorical(training_set.labels);
testLabels = categorical(test_set.labels);

%% Train the decision tree classifier
tree_model = fitctree(trainData, trainLabels,MinLeafSize=5,SplitCriterion="twoing");
%% Visualize the progress during training
view(tree_model, 'Mode', 'graph');
%% Predict labels for test data
tree_pred = predict(tree_model, testData);

%% Convert predicted labels to categorical
tree_pred = categorical(tree_pred);

%% Calculate accuracy
accuracy = sum(tree_pred == testLabels) / numel(testLabels);
fprintf('Decision Tree Accuracy: %f\n', accuracy);

%% Plot Confusion Matrix
confusion_tree = confusionmat(testLabels, tree_pred);
confusionchart(confusion_tree);
title('Confusion Matrix for Decision Tree');
