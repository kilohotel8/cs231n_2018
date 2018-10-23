import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        d_margin_score = 1
        d_margin_correct_score = -1
        d_score_W = X[i]
        d_correct_score_W = X[i]
        d_margin_W = d_margin_score * d_score_W
        d_margin_W_correct = d_margin_correct_score * d_correct_score_W
        dW[:,j] += d_margin_W
        dW[:,y[i]] += d_margin_W_correct
    

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_scores = scores[np.arange(len(scores)),y].reshape(500,1)
  temp_loss = scores - correct_scores + 1
  temp_loss[temp_loss<0] = 0
  temp_loss[np.arange(len(temp_loss)),y] = 0
  loss = np.sum(temp_loss,axis = 1)
  loss = np.mean(loss)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  d_loss_scores = np.zeros(temp_loss.shape)
  d_loss_scores[temp_loss > 0] = 1
  num_train = X.shape[0]
  incorrect = np.sum(d_loss_scores, axis=1)
  d_loss_scores[np.arange(X.shape[0]), y] = -incorrect
  dW = np.dot(X.T,d_loss_scores)
  dW /= X.shape[0]
  dW += 2*reg*W
  
  
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  loss += reg * np.sum(W * W)

  return loss, dW
