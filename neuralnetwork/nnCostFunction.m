function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % number of training examples
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

% obtain version of y where labels (1-8) are represented by a 1x8 matrix
% corresponding to the 8 possible output units of the neural network
% comprised of 0's and one 1 indicated the predicted value
y2 = zeros(m, num_labels); 
for c = 1:m
    y2(c, y(c)) = 1;   
end

% calculate cost function for all training examples
tempJ = 0;
for t = 1:m
    a1 = X(t,:);
    a2 = sigmoid([1 a1] * Theta1');
    a3 = sigmoid([1 a2] * Theta2');
    tempJ = tempJ + (-y2(t,:) * log(a3)' - (1-y2(t,:)) * log(1-a3)');
end

J = 1/m * tempJ;
% add regularization to prevent overfitting of the data (prevent the neural
% network from contorting the hypothesis to capture all the data points as 
% opposed to finding a smooth function that fits the overall trend of the 
% data points)
reg = lambda / (2*m);
sum1 = sum(sum(Theta1(:,2:end) .^2));
sum2 = sum(sum(Theta2(:,2:end) .^2));
J = J + reg*(sum1 + sum2);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. Return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

% backpropagation algorithm
Delta1 = zeros(size(Theta1_grad));
Delta2 = zeros(size(Theta2_grad));
for i=1:m
    a1 = X(i,:);
    a2 = sigmoid([1 a1] * Theta1');
    a3 = sigmoid([1 a2] * Theta2');
    d3 = a3 - y2(i,:);
    d2 = Theta2'* d3' .* ([1 a2]'.* (1-[1 a2])');
    Delta2 = Delta2 + d3' * [1 a2];
    Delta1 = Delta1 + d2(2:end) * [1 a1];
end
% make sure to only regularize non-bias terms (omit first column)
Theta1_grad(:,2:end) = 1/m * Delta1(:,2:end) + lambda/m * Theta1(:,2:end);
Theta1_grad(:, 1) = Delta1(:,1)/m;
Theta2_grad(:,2:end) = 1/m * Delta2(:,2:end) + lambda/m * Theta2(:,2:end);
Theta2_grad(:, 1) = Delta2(:,1)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients to use in minimization function
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
