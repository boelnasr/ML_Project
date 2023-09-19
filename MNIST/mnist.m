close all;
clear all;
clc;

% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool;
end
%% Load MNIST dataset
data = load ('mnist.mat');

%% Define both the trainiing and testing sets and there sizes:

training_set = data.training;
test_set = data.test;
size_training_set = data.training.count;
size_testing_set = data.test.count;
train_Labels = gpuArray(training_set.labels);
test_Labels =gpuArray( test_set.labels);
% Reshape images into vectors
trainData =gpuArray(reshape(training_set.images, [], size(training_set.images, 3))');
testData =gpuArray(reshape(test_set.images, [], size(test_set.images, 3))');

%% K-nearest Neighbors

tic
% Train k-NN
knn_model = fitcknn(trainData, train_Labels, 'NumNeighbors', 3, ...
    "Distance","correlation", "DistanceWeight","squaredinverse");
% Predict
knn_pred = predict(knn_model, testData);
% calculate accuracy
knn_accuracy = sum(knn_pred == test_Labels) / numel(test_Labels);
fprintf('k-NN Accuracy: %f\n', knn_accuracy);
time_knn = toc
%% Plot Confusion Matrix
test_Labels=gather(test_Labels);
knn_pred=gather(knn_pred);
confusionKNN = confusionmat(test_Labels, knn_pred);
confusionchart(confusionKNN);
title('Confusion Matrix for k-NN');
