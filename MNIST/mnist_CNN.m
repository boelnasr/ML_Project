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
    'MaxEpochs', 10, ...              % Increase the number of epochs
    'InitialLearnRate', 0.001, ...    % Adjust the initial learning rate
    'MiniBatchSize', 128, ...         % Experiment with different batch sizes
    'ValidationData', {testImages, categorical(testLabels)}, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'parallel', ...
    'Shuffle', 'every-epoch', ...     % Shuffle the data every epoch
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...     % Decrease the learning rate every 5 epochs
    'LearnRateDropFactor', 0.5, ...   % Reduce the learning rate by half
    'L2Regularization', 0.0001);      % Apply L2 regularization

%% Train the CNN
[net, info] = trainNetwork(trainImages, categorical(trainLabels), lgraph, options);
% Evaluate the trained network
pred = classify(net, testImages);
accuracy = sum(pred == categorical(testLabels)) / numel(testLabels);

fprintf('Test accuracy: %f\n', accuracy);
%% Training info
% Your ValidationLoss array
validationLoss = [trainingInfo.ValidationLoss];  % Replace with your actual validation loss array

% Your TrainingLoss array
trainingLoss = [trainingInfo.TrainingLoss];  % Replace with your actual training loss array

% Find non-NaN indices
validIndices = ~isnan(validationLoss);

% Remove NaN values
validationLoss = validationLoss(validIndices);

% Initialize the figure
figure;

% Subplot 1: Training Loss
subplot(2, 1, 1);  % 1 row, 2 columns, first subplot
plot(trainingLoss, 'LineWidth', 2);  % Assumes your training loss does not contain NaN values
xlabel('Iteration', 'Interpreter', 'latex','FontSize',15, 'FontName','Times New Roman');
ylabel('Loss', 'Interpreter', 'latex','FontSize',15, 'FontName','Times New Roman');
legend('Training Loss', 'Interpreter', 'latex','FontSize',15, 'FontName','Times New Roman');
title('Training Loss Curve', 'Interpreter', 'latex','FontSize',20, 'FontName','Times New Roman');
grid on;

% Subplot 2: Validation Loss
subplot(2, 1, 2);  % 1 row, 2 columns, second subplot
plot(validationLoss, 'LineWidth', 2);
xlabel('Iteration', 'Interpreter', 'latex','FontSize',15, 'FontName','Times New Roman');
ylabel('Loss', 'Interpreter', 'latex','FontSize',15, 'FontName','Times New Roman');
legend('Training Loss', 'Interpreter', 'latex','FontSize',15, 'FontName','Times New Roman');
title('Training Loss Curve', 'Interpreter', 'latex','FontSize',20, 'FontName','Times New Roman');
grid on;
%% Calculate the confusion matrix
C = confusionmat(categorical(testLabels), pred);
% Visualize the network architecture
figure;
plot(lgraph);
title('CNN Architecture Visualization','Interpreter', 'latex','FontSize',20, 'FontName','Times New Roman');
% Plot the confusion matrix
figure;
confusionchart(C);
title('Confusion Matrix for CNN');
% Calculate precision, recall, and F1-score
nClasses = size(C, 1);  % Assuming C is a square matrix
precision = zeros(1, nClasses);
recall = zeros(1, nClasses);
f1Score = zeros(1, nClasses);

for i = 1:nClasses
    TP = C(i, i);
    FP = sum(C(:, i)) - TP;
    FN = sum(C(i, :)) - TP;
    
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Display the precision, recall, and F1-score
fprintf('Precision per class: %s\n', mat2str(precision, 4));
fprintf('Recall per class: %s\n', mat2str(recall, 4));
fprintf('F1-score per class: %s\n', mat2str(f1Score, 4));
