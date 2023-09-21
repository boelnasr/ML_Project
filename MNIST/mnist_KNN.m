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
%% Compute PCA on the training data
[coeff, score, latent] = pca(gather(trainData));

% Determine the number of principal components to use
explainedVariance = cumsum(latent) / sum(latent);
numComponents = find(explainedVariance > 0.95, 1); % This captures 95% of the total variance

% Project the training data into the PCA space
trainDataPCA = gpuArray(score(:, 1:numComponents));

% Project the test data into the PCA space
meanTrain = mean(trainData, 1);
testDataPCA = gpuArray((gather(testData) - meanTrain) * coeff(:, 1:numComponents));
%% Explained Variance Plot
figure;
subplot(1,2,1); % This allows for 1 row, 2 columns in the figure. We'll plot in the first column.
stem(latent/sum(latent), 'LineWidth', 2);  % This plots the proportion of variance for each component
xlabel('Principal Component', 'Interpreter', 'latex', 'FontSize', 25);
ylabel('Explained Variance', 'Interpreter', 'latex', 'FontSize', 25);
title('Explained Variance per Principal Component', 'Interpreter', 'latex', 'FontSize', 30);
grid on;

% Cumulative Explained Variance Plot
subplot(1,2,2); % This plots in the second column.
plot(explainedVariance, 'LineWidth', 2);
xlabel('Number of Principal Components', 'Interpreter', 'latex', 'FontSize', 25);
ylabel('Cumulative Explained Variance', 'Interpreter', 'latex', 'FontSize', 25);
title('Cumulative Explained Variance', 'Interpreter', 'latex', 'FontSize', 30);
grid on;

% Highlighting the point where 95% variance is explained
hold on;
plot(numComponents, explainedVariance(numComponents));
legend('Cumulative Variance','95% Explained', 'Interpreter', 'latex', 'FontSize', 15);
hold off;


%% Cumulative Explained Variance Plot
subplot(1,2,2); % This plots in the second column.
plot(explainedVariance, 'LineWidth', 2);
xlabel('Number of Principal Components', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('Cumulative Explained Variance', 'Interpreter', 'latex', 'FontSize', 15);
title('Cumulative Explained Variance', 'Interpreter', 'latex', 'FontSize', 20);
grid on;

% Highlighting the point where 95% variance is explained
hold on;
plot(numComponents, explainedVariance(numComponents), 'ro');
legend('Cumulative Variance','95% Explained', 'Interpreter', 'latex', 'FontSize', 15);
hold off;
%% K-nearest Neighbors

tic
% Train k-NN using the PCA-transformed training data
knn_model = fitcknn(trainDataPCA, train_Labels, 'NumNeighbors', 5, ...
    "Distance","correlation", "DistanceWeight","squaredinverse");

% Predict using the PCA-transformed test data
knn_pred = predict(knn_model, testDataPCA);

% Calculate accuracy
knn_accuracy = sum(knn_pred == test_Labels) / numel(test_Labels);
fprintf('k-NN Accuracy (after PCA): %f\n', knn_accuracy);
time_knn = toc
%% Plot Confusion Matrix
test_Labels=gather(test_Labels);
knn_pred=gather(knn_pred);
confusionKNN = confusionmat(test_Labels, knn_pred);
figure;
confusionchart(confusionKNN);
title('Confusion Matrix for k-NN');
% Calculate precision, recall, and F1-score
C= confusionKNN;
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


