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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  
  for i in range(num_train):
    incorrect_class_score_sum = 0
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] -= X[i, :] # derivative wrt -z_yi
        loss -= scores[i, j] # loss for correct class
      else:
        incorrect_class_score_sum += np.exp(scores[i, j])
    
    for j in range(num_classes):
      if j != y[i]:
        dW[:, j] += X[i, :] * np.exp(scores[i, j]) / incorrect_class_score_sum # derivative wrt log sum(exp(z_j))
    
    loss += np.log(incorrect_class_score_sum) # loss for incorrect classes
  
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization to dW  
  dW += 2 * reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  scores = np.exp(scores)
  
  incorrect_classes_mask = np.ones_like(scores)
  incorrect_classes_mask[np.arange(num_train), y] = 0

  # sum over all j exp(f_j)
  incorrect_classes_score_sum = np.sum(scores * incorrect_classes_mask, axis=1)
  # exp(f_yi)
  correct_classes_scores = scores[np.arange(num_train), y]

  # exp(f_yi) / sum(exp(f_j)) 
  loss = correct_classes_scores / incorrect_classes_score_sum
  loss = -1 * np.log(loss)
  loss = np.sum(loss)

  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  dW_mask = np.copy(scores)
  dW_mask = dW_mask / incorrect_classes_score_sum.reshape((-1, 1))
  dW_mask[np.arange(num_train), y] = -1

  dW = X.T.dot(dW_mask)
  dW /= num_train
  
  # Add regularization to dW  
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

