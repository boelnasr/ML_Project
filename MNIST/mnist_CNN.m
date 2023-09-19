%% Clear all variables, close all figures and clear the console
clear all;
close all;
clc;

% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool;
end

%% Load MNIST dataset
data = load('mnist.mat');  % Make sure you have mnist.mat in the working directory

% Prepare Training Data
trainImages = data.training.images;
trainImages = reshape(trainImages, [28, 28, 1, size(trainImages, 3)]);
trainImages = double(trainImages) / 255;
trainLabels = data.training.labels;

% Prepare Test Data
testImages = data.test.images;
testImages = reshape(testImages, [28, 28, 1, size(testImages, 3)]);
testImages = double(testImages) / 255;
testLabels = data.test.labels;

%% Define CNN architecture using layerGraph
lgraph = layerGraph([
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer]);

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...              % Increase the number of epochs
    'InitialLearnRate', 0.001, ...    % Adjust the initial learning rate
    'MiniBatchSize', 128, ...         % Experiment with different batch sizes
    'ValidationData', {testImages, categorical(testLabels)}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'parallel', ...
    'Shuffle', 'every-epoch', ...     % Shuffle the data every epoch
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...     % Decrease the learning rate every 5 epochs
    'LearnRateDropFactor', 0.5, ...   % Reduce the learning rate by half
    'L2Regularization', 0.0001);      % Apply L2 regularization

%% Train the CNN
net = trainNetwork(trainImages, categorical(trainLabels), lgraph, options);
% Evaluate the trained network
pred = classify(net, testImages);
accuracy = sum(pred == categorical(testLabels)) / numel(testLabels);

fprintf('Test accuracy: %f\n', accuracy);
%% Calculate the confusion matrix
C = confusionmat(categorical(testLabels), pred);
% Visualize the network architecture
figure;
plot(lgraph);
title('CNN Architecture Visualization');
% Plot the confusion matrix
figure;
confusionchart(C);
title('Confusion Matrix for CNN');
