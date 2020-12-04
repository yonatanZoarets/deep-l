import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def gradient_check(f, x):
    """ 
    Gradient test.pred for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to test.pred the gradient at
    """ 
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.
        ### YOUR CODE HERE:
        x[ix]+=h
        y1=f(x)[0]
        x[ix]-=2*h
        y2=f(x)[0]
        numerical_grad=(y1-y2)/(2*h)
        x[ix]+=h
        print(numerical_grad)

        ### END YOUR CODE
        #raise notImplemented
        # Compare gradients
        reldiff = abs(numerical_grad - grad[ix]) / max(1, abs(numerical_grad), abs(grad[ix]))
        if reldiff > 1e-5:
            print ("Gradient test.pred failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numerical_grad))
            return
    
        it.iternext() # Step to next index

    print ("Gradient test.pred passed!")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print ("Running sanity checks...")
    gradient_check(quad, np.array(123.456))      # scalar test
    gradient_check(quad, np.random.randn(3,))    # 1-D test
    gradient_check(quad, np.random.randn(4,5))   # 2-D test
    print ("")

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()
