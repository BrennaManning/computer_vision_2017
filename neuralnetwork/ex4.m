%% Machine Learning Neural Network - Classifying Facial Expressions/Emotions
% Marissa Okoli

%  A portion of this code comes from Andrew Ng's online course on machine
%  learing on Coursera. I modified the files below:
%  ------------
%
%     ex4.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%

%% Initialization
clear ; close all; clc


%% Setup the parameters for neural network
input_layer_size  = 576;  % 24x24 Input Images of Faces
hidden_layer_size = 36;   % 36 hidden units
num_labels = 8;          % 8 labels: anger, contempt, disgust, fear, happy, 
                         % neutral, sad surprised
                         

%% =========== Part 1: Loading and Visualizing Data =============
%  Load and visualize a subset of the data (from the exercise). Create
%  matrix out of the images from the database.
% 
%Note subfolders were added to search path
%40 in anger
%15 in contempt
%47 in disgust
%22 in fear
%55 in happy
%85 in neutral
%24 in sad
%63 in surprised

X = zeros(351, 576);
y = zeros(351, 1);
y(1:40) = 1; % anger
y(41:55) = 2; % contempt
y(56:102) = 3; % disgust
y(103:124) = 4; % fear
y(125:179) = 5; % happy
y(180:264) = 6; % neutral
y(265:288) = 7; % sadness
y(289:end) = 8; %surprise

for t = 1:size(X,1)
    photo = imread(strcat('expr',int2str(t),'.png'));
    X(t,:) = photo(:)';
end


% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================
%  Initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training NN ===================
%  Use "fmincg" (also could have used gradient descent), which 
%  is a function which works similarly to "fminunc". These
%  advanced optimizers are able to train cost functions efficiently as
%  long as they are provided with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%set maximum iterations for neural network
options = optimset('MaxIter', 30000);

%  Regularization term that helps prevent overfitting of the data
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 4: Visualize Weights =================
%  "Visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data (from the exercise).

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 5: Implement Predict =================
%  After training the neural network,  use it to predict
%  the labels. This allows computation of the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


