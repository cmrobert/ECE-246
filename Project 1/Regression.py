import numpy as np
class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns: - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            X_out[:,0] = 1
            X_out[:,1:m+1] = X
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_out[:,0] = 1
            for i in range(1,m+1):
                 temp = []
                 temp = np.power(X, i)
                 X_out[:,i] = temp.ravel()

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        #print(self.w.shape)
        #print(X.shape)
        L = self.reg
        #print(L)
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #
            #print(self.w.shape)
            temp1 = []
            temp2 = []
            temp3 = []
            temp4 =[]
            tmp1 = []
            tmp2 = []
            
            
            if X.shape[1] == 1:
                X_out= np.zeros((N,m+1))
                X_out[:,0] = 1
                X_out[:,1:m+1] = X
                X = X_out
            
            xt = np.transpose(X)
            temp1 = np.matmul(xt,X)
            temp2 = np.matmul(temp1,self.w)
            temp3 = np.matmul(xt,y)
            temp4 = self.reg*self.w
           
            grad = temp2[:,0] + temp4[:,0] - temp3
            grad = 2*grad/N
            tmp1 = np.matmul(X,self.w)
            tmp2 = tmp1[:,0] - y
            loss = np.matmul(np.transpose(tmp2),tmp2)
            loss = loss/N
            reg_term = self.reg*np.matmul(np.transpose(self.w),self.w)/2
            loss = loss + reg_term
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            temp1 = []
            temp2 = []
            temp3 = []
            temp4 =[]
            tmp1 = []
            tmp2 = []
      
            
            xt = np.transpose(X)
            temp1 = np.matmul(xt,X)
            temp2 = np.matmul(temp1,self.w)
            temp3 = np.matmul(xt,y)
            temp4 = self.reg*self.w
           
            grad = temp2[:,0] + temp4[:,0] - temp3
            grad = 2*grad/N
            tmp1 = np.matmul(X,self.w)
            tmp2 = tmp1[:,0] - y
            loss = np.matmul(np.transpose(tmp2),tmp2)
            loss = loss/N
            reg_term = self.reg*np.matmul(np.transpose(self.w),self.w)/2
            loss = loss + reg_term
            
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
           
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
                X_batch = np.zeros((batch_size,1))
                y_batch = np.zeros(batch_size)
                ind = []
                rng = []
                rng = np.arange(N)
                ind = np.random.choice(rng,batch_size)
                
                for i in range(0,batch_size):
                    X_batch[i,0] = X[ind[i]]
                    y_batch[i] = y[ind[i]]
                        
                    
                    
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss = 0.0
                grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
                '''
                w = self.w
                temp = []
                gtemp = []
                for i in range(0,batch_size):
                    temp = np.power((X_batch[i,:]*w[1] + w[0] - y_batch[i]),2)
                    loss = loss + temp
                    gtemp = 2*(X_batch[i,:]*w[1] + w[0] - y_batch[i])*X_batch[i,:]
                    grad = grad + gtemp
                loss = loss/batch_size
                grad = grad/batch_size
                '''
                #print(self.w.shape)
                w = self.w
                grad0 = 0
                grad1 = 0
                for i in range(0,batch_size):
                    temp = np.power(np.dot(X_batch[i,:],w[1]) + w[0] - y_batch[i],2)
                    loss = loss + temp
                    gtemp0 = 2*(X_batch[i,:]*w[1] +w[0] - y_batch[i])
                    grad0 = grad0 + gtemp0
                    gtemp1 = 2*(X_batch[i,:]*w[1] +w[0] - y_batch[i])*X_batch[i,:]
                    grad1 = grad1 + gtemp1


                loss = loss/N
                grad[0] = grad0/N
                grad[1] = grad1/N
                
                
                
                
                
                self.w = self.w - (eta * grad)
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        if m==1:
            
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #
            from numpy.linalg import inv
            if d == 1:
                X_out= np.zeros((N,m+1))
                X_out[:,0] = 1
                X_out[:,1] = X[:,0]
                X = X_out
            
            
            loss = 0.0
            x2 = np.matmul(np.transpose(X),X)
            #print(x2)
            inv = inv(x2)
            #print(inv)
            times = np.matmul(np.transpose(X),y)
            self.w = np.matmul(inv,times)
           
            
            #print(X.shape)
            w = self.w
            tmp1 = []
            tmp2 = []
            tmp1 = np.matmul(X,self.w)
           
            tmp2 = tmp1 - y
            loss = np.matmul(np.transpose(tmp2),tmp2)
            loss = loss/N
            reg_term = self.reg*np.matmul(np.transpose(self.w),self.w)
            reg_term = reg_term/2
            loss = loss + reg_term
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            from numpy.linalg import inv
            temp1 = []
            temp2 = []
            temp3 = []
            temp4 = []
            loss = 0.0
            xt = np.transpose(X)
            #print(X.shape)
            temp1 = np.matmul(xt,X)
            #print(temp1.shape)
            
            temp2 = temp1 + self.reg*np.identity(self.m+1)
            #print(temp4)
            temp3 = np.matmul(xt,y)
            temp4 = inv(temp2)
            #print(temp4)
            self.w = np.matmul(temp4,temp3)

            #print(X.shape)
            w = self.w
            tmp1 = []
            tmp2 = []
            tmp1 = np.matmul(X,self.w)
            tmp2 = tmp1 - y
            loss = np.matmul(np.transpose(tmp2),tmp2)
            loss = loss/N
            reg_term = self.reg*np.matmul(np.transpose(self.w),self.w)
            reg_term = reg_term/2
            loss = loss + reg_term
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #

        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            #print(self.w)
            N,d = X.shape
            from numpy.linalg import inv
            X_out= np.zeros((N,m+1))
            X_out[:,0] = 1
            X_out[:,1] = X[:,0]
            
            for i in range(0,X.shape[0]):
                #print(X_out[i,:])
                y_pred[i] = np.matmul(self.w,X_out[i,:])
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            for i in range(0,X.shape[0]):
                y_pred[i] = np.matmul(self.w,X[i,:])
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
          
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred