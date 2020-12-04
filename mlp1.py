import math

import numpy as np

import loglinear as ll


STUDENT={'name': 'Yonatan Zoarets',
         'ID': '207596818'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    W,b,U,b_tag=params
    temp=np.add(np.dot(np.tanh(np.add(np.dot(x,W),b)), U), b_tag)
    probs=ll.softmax(temp)/np.sum(ll.softmax(temp))
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    [W,b,U,b_tag]=params

    y_des = [0 for _ in range(len(b_tag))]
    y_des[y] = 1

    y_pred = classifier_output(x, params)
    loss = (-np.log(y_pred[y]))
    par=[W,b]

    gb_tag=y_pred-y_des #as we did in train loglin
    gU=np.array([np.tanh(np.add(np.dot(x, W), b))]).T.dot([gb_tag])
    gb=np.arctanh(np.subtract((np.subtract(y_pred,b_tag)),(np.subtract(y_des,b_tag))))#gradients what make the Wx+b accurater
    # np.tanh(np.add(np.dot(x, W), b))
    gW = np.array([x]).T.dot([gb])
    # print( loss, ["gb_tag",gb_tag],"gb",gb,gW)

    return loss, [gW, gb,gU,gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.zeros((in_dim,out_dim))
    b = np.zeros(out_dim)
    b_tag=np.zeros(hid_dim)
    U=np.zeros((hid_dim,out_dim))

    params = [W,b,U,b_tag]
    return params

