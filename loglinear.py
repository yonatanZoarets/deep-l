import numpy as np

STUDENT={'name': 'Yonatan Zoarets',
         'ID': '207596818'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    x=np.array(x)
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    c=-max(x)
    e=np.exp(x+c)
    all=np.sum(e)
    x=e/all #as we learned

    return x
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    # YOUR CODE HERE.
    probs = softmax(np.add(np.dot(x,W),b))/np.sum(softmax(np.add(np.dot(x,W),b)))# softmax(Wx+b)
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    # YOU CODE HERE
    x=np.array(x)

    yx = np.array([0 for _ in range(len(b))])
    yx[y] = 1
    print("y_des",yx)
    y_pred=classifier_output(x,params)
    # print("l=","0.001*(",yi, "-", "(",m,"*",xi,"+", c,")) ** 2 = ",l)
    loss = (-np.log(y_pred[y]))
    gb=y_pred-yx
    gW=np.array([x]).T.dot([gb])# we didnt know about T function
    # print(gb,np.array([x]),"gW",gW,"/gW")
    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim,out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.


    test1 = softmax(np.array([1,2])) #its [0.3333333 , 0.6666667]
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4) #W is matrix, b is vector

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],3,[W,b])
        return loss,grads[0]


    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],1,[W,b])
        return loss,grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        print("W",W ,"b",b)

        gradient_check(_loss_and_b_grad,b)
        gradient_check(_loss_and_W_grad,W)


    
