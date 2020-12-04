
import numpy as np
import loglinear as ll
import random

from test_pred_code import read_data
from utils import text_to_bigrams

STUDENT={'name': 'Yonatan Zoarets',
         'ID': '207596818'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return np.array(features)

def accuracy_on_dataset(dataset, params):
    W,b=params
    good = bad = 0.0
    for label, features in dataset:
        y_des = [0 for _ in range(len(b))]
        y_des[label]=1
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if np.sum(np.fabs(np.subtract(ll.classifier_output(features,params),y_des)))<1e-2: #doing x*params because the shape. 1e-3 close enough
            good+=1
        else:
            bad+=1
        pass
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label# convert the label to number if needed.
            # YOUR CODE HERE
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            params =np.subtract(params,np.multiply(grads,learning_rate))  #must to do minus
            print(ll.classifier_output(x,params))

             #gradients computed previous file, now the sgd is the train
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print (I, train_loss, train_accuracy, dev_accuracy)
    return params


def pairs(length):

    return [(random.randint(0,3),np.random.randint(18,size=3) ) for i in range(length)] #size=3 match all together

if __name__ == '__main__':
    # YOUR CODE HERE

    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    in_dim=3
    out_dim=4
    # matching to the shape, because what we defined previous file-if we want to change- we will do change also in the previous
    dev_data=pairs(5)
    train_data=pairs(5)
    learning_rate=1.05
    num_iterations=5
    params=ll.create_classifier(in_dim,out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
