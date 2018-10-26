import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_data = np.shape(X)[0]
  dimension = np.shape(X)[1]
  num_class = np.shape(W)[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_data):
    z = np.dot(X[i], W).reshape(num_class,1)
    exp_z = np.exp(z)
    softmax = exp_z / np.sum(exp_z)
    loss += -np.log(softmax[y[i]])
    for j in range(num_class):
        if j == y[i]:
            dW[:,j] += X[i]*(softmax[j] - 1)
        else:
            dW[:,j] += X[i]*softmax[j]
  
  dW /= num_data
  dW += 2*reg*W
  loss /= num_data
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_data = np.shape(X)[0]
  num_class = np.shape(W)[1]
  y_onehot = np.zeros((num_data,num_class))
  y_onehot[np.arange(num_data),y] = 1

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  z = np.dot(X, W)
  exp_z = np.exp(z)
  exp_sum = np.sum(exp_z, axis=1).reshape(num_data,1)
  softmax = exp_z/exp_sum
  temp_loss = -np.log(softmax[np.arange(num_data),y])
  loss = np.mean(temp_loss)
  error = softmax - y_onehot
  dW = np.matmul(X.T,error)
  dW /= num_data
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += np.sum(reg*W*W)
  dW += 2*reg*W
  return loss, dW

