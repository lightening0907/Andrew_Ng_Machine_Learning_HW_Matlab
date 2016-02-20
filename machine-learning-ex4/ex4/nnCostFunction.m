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
m = size(X, 1);
X_i = [ones(m,1),X];         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
z2 = X_i*(Theta1.');
a2=sigmoid(z2);
a2_i=[ones(size(a2,1),1),a2];
z3 = a2_i*(Theta2.');
h_theta = sigmoid(z3);
J_ur = 0; %unregularized cost function
for i =1:num_labels
   J_ur =J_ur+ (sum(-log(h_theta(:,i)).*(y==i)-(1-(y==i)).*log(1-h_theta(:,i)))/m);
end
Jr = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);
J = J_ur+Jr;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
d_3 = zeros(m,num_labels);
d_2 = zeros(m,hidden_layer_size+1);
a = 1:num_labels;
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t=1:m
    d_3(t,:)=h_theta(t,:)-(a==y(t));
    gz = (sigmoidGradient([1,z2(t,:)]));
    d_2(t,:) = (d_3(t,:)*(Theta2)).*gz;
    Delta1 = Delta1 + d_2(t,2:end).'*X_i(t,:);
    Delta2 = Delta2 + d_3(t,:).'*a2_i(t,:);
end

Theta1_grad_nreg = Delta1/m;
Theta2_grad_nreg = Delta2/m;

Theta1_grad = Theta1*lambda/m;
Theta1_grad(:,1)=0;
Theta1_grad = Theta1_grad + Theta1_grad_nreg;

Theta2_grad = Theta2*lambda/m
Theta2_grad(:,1)=0;
Theta2_grad = Theta2_grad + Theta2_grad_nreg;
   % for k = 1:num_labels
   % d_3(t,k)=(h_theta(t,k) - y(t)==k);
   % end
    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
