from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout=None):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # layer 1 pass
    layer_1_out = X.dot(W1) + b1
    # reLU activation
    A = np.copy(layer_1_out)
    A[layer_1_out <= 0] = 0
    # apply dropout
    if dropout is not None:
        dropout_probs = np.random.random(A.shape[1])
        dropout_mask = np.zeros_like(dropout_probs)
        dropout_mask[dropout_probs > dropout] = 1.
        A *= dropout_mask
        # scale outputs by prob of being active during training 
        # to avoid scaling at test time
        A *= 1 - dropout 
    # layer 2 pass
    layer_2_out = A.dot(W2) + b2
    scores = layer_2_out

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    
    # loss is sum of exp(correct_class_score) / sum(exp(incorrect_class_scores))
    
    # shift numbers so max is zero 
    # to improve the numerical stability of the computation
    scores -= np.max(scores, axis=1, keepdims=True) 
    
    # unnormalized probabilities
    probs = np.exp(scores)

    # sum of unnormalized probabilities per example
    prob_sums = np.sum(probs, axis=1, keepdims=True)
    
    # normalized probabilities
    probs /= prob_sums 
    
    # loss will be sum of - log of correct class probabilities for each example
    correct_class_probs = probs[np.arange(N), y]
    log_correct_class_probs = -1 * np.log(correct_class_probs)
    loss = np.sum(log_correct_class_probs)
    loss /= N
    
    # add regularization to loss
    loss += reg * (np.sum(W1 * W1) + np.sum(b1 * b1) + np.sum(W2 * W2) + np.sum(b2 * b2))
  
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
 
    # dL/dZ2
    dZ2 = np.copy(probs)
    dZ2[range(N), y] -= 1
    
    # dZ2/dW2  
    dZ2_dW2 = A
    
    # dL/dW2 = dZ2/dW2 .T dot dL/dZ2
    dW2 = np.dot(dZ2_dW2.T, dZ2)
    dW2 /= N
    
    # dZ2/db2
    dZ2_db2 = np.ones(N)
    
    # dL/db2 = dZ2/db2 .T dot dL/dZ2
    db2 = np.dot(dZ2_db2.T, dZ2)
    db2 /= N
    
    if dropout is not None:
        # dZ2_dD2
        dZ2_dD = W2
        
        # dL/dD
        dD = np.dot(dZ2, dZ2_dD.T)
        
        # dD/dA
        dD_dA = dropout_mask
        dA = dD * dD_dA
    else:
        # dZ2/dA
        dZ2_dA = W2

        # dL/dA
        dA = np.dot(dZ2, dZ2_dA.T)
    
    # dA/dZ1
    dA_dZ2 = np.zeros_like(layer_1_out)
    dA_dZ2[layer_1_out > 0] = 1
    
    # dL/dZ1
    dZ1 = dA * dA_dZ2
    
    # dZ1/dW1
    dZ1_dW1 = X
    
    # dL/dW = dZ1/dW1 * dL/dZ1
    dW1 = np.dot(dZ1_dW1.T, dZ1)
    dW1 /= N
    
    # dZ1/db1
    dZ1_db1 = np.ones(N).reshape((-1, 1))
    
    # dL/db1
    db1 = np.squeeze(np.dot(dZ1_db1.T, dZ1))
    db1 /= N
    
    # Add regularization
    dW2 += 2 * reg * W2
    db2 += 2 * reg * b2
    dW1 += 2 * reg * W1
    
    db1 += 2 * reg * b1
    
    assert dW2.shape == W2.shape
    assert db2.shape == b2.shape
    assert dW1.shape == W1.shape
    assert db1.shape == b1.shape
    
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False, dropout=None):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    - dropout: probability of dropping a hidden unit
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      train_batch_idx = np.random.choice(num_train, batch_size)
      X_batch = X[train_batch_idx, :]
      y_batch = y[train_batch_idx]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout=None)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
      self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
      self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']
      self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']

        
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


