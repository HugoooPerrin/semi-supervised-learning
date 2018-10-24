#!/usr/bin/env python
# -*- coding: utf-8 -*-



#=========================================================================================================
#================================ 0. MODULE


import numpy as np
import math
from numpy import linalg

import sklearn
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph

import scipy.optimize as sco

from itertools import cycle, islice


#=========================================================================================================
#================================ 1. ALGORITHM


class LapRLS(object):

    def __init__(self, n_neighbors, kernel, lambda_k, lambda_u, 
                 learning_rate=None, n_iterations=None, solver='closed-form'):
        """
        Laplacian Regularized Least Square algorithm

        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        Learning_rate: float
            Learning rate of the gradient descent
        n_iterations : integer
        solver : string ('closed-form' or 'gradient-descent' or 'L-BFGS-B')
            The method to use when solving optimization problem
        """
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.solver = solver
        

    def fit(self, X, X_no_label, Y):
        """
        Fit the model
        
        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """
        # Storing parameters
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        self.Y = np.concatenate([Y, np.zeros(u)])
                
        # Memory optimization
        del X_no_label
        
        # Building adjacency matrix from the knn graph
        print('Computing adjacent matrix', end='...')
        W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity')
        W = (((W + W.T) > 0) * 1)
        print('done')

        # Computing Graph Laplacian
        print('Computing laplacian graph', end='...')
        L = np.diag(W.sum(axis=0)) - W
        print('done')

        # Computing K with k(i,j) = kernel(i, j)
        print('Computing kernel matrix', end='...')
        K = self.kernel(self.X)
        print('done')

        # Creating matrix J (diag with l x 1 and u x 0)
        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))
        
        if self.solver == 'closed-form':
            
            # Computing final matrix
            print('Computing final matrix', end='...')
            final = (J.dot(K) + self.lambda_k * l * np.identity(l + u) + ((self.lambda_u * l) / (l + u) ** 2) * L.dot(K))
            print('done')
        
            # Solving optimization problem
            print('Computing closed-form solution', end='...')
            self.alpha = np.linalg.inv(final).dot(self.Y)
            print('done')
            
            # Memory optimization
            del self.Y, W, L, K, J
            
        elif self.solver == 'gradient-descent':
            """
            If solver is Gradient-descent then a learning rate and an iteration number must be provided
            """
            
            print('Performing gradient descent...')
            
            # Initializing alpha
            self.alpha = np.zeros(n)

            # Computing final matrices
            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(L)).dot(K)

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
                        
            # Memory optimization
            del self.Y, W, L, K, J
        
            for i in range(self.n_iterations + 1):
                
                # Computing gradient & updating alpha
                self.alpha -= self.learning_rate * RLS_grad(self.alpha)
                
                if i % 50 == 0:
                    print("\r[%d / %d]" % (i, self.n_iterations) ,end = "")
                    
            print('\n')
        
        elif self.solver == 'L-BFGS-B':
            
            print('Performing L-BFGS-B', end='...')
            
            # Initializing alpha
            x0 = np.zeros(n)

            # Computing final matrices
            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(L)).dot(K)

            def RLS(alpha):
                return np.squeeze(np.array((1 / l) * (self.Y - J.dot(K).dot(alpha)).T.dot((self.Y - J.dot(K).dot(alpha))) \
                        + self.lambda_k * alpha.dot(K).dot(alpha) + (self.lambda_u / n ** 2) \
                        * alpha.dot(K).dot(L).dot(K).dot(alpha)))

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
            
            self.alpha, _, _ = sco.fmin_l_bfgs_b(RLS, x0, RLS_grad, args=(), pgtol=1e-30, factr =1e-30)
            
            print('done')
                                    
        # Finding optimal decision boundary b using labeled data
        new_K = self.kernel(self.X, X)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == Y) / len(predictions))

        bs = np.linspace(0, 1, num=101)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]
        

    def predict(self, Xtest):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : ndarray shape (n_samples, )
            Predicted labels for Xtest
        """

        # Computing K_new for X
        new_K = self.kernel(self.X, Xtest)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        predictions = np.array((f > self.b) * 1)
        return predictions
    

    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        accuracy = sum(predictions == Ytrue) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))