# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:53:35 2020

@author: Steven
"""


import numpy as np

import tensorflow as tf

import os

os.chdir('D:\\Kaggle\\code')

import math

from formatData import *


learning_rate = 1
epochs = 1
batchSize = 100


# create placeholders for data

x_image = tf.placeholder(tf.float32, [None, 64, 64, 30])

x_clin = tf.placeholder(tf.float32, [None, clinDataDim] )

y = tf.placeholder(tf.float32, [None])




# This function creates a layer that convoles and then pools.


def createConvLayer(inputData, numInputChannels, numOutChannels, filterShape, poolShape, name):
    # setup the filter input shape for tf.nn.conv_2d
    convFiltShape = [filterShape[0], filterShape[1], numInputChannels,
                      numOutChannels]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(convFiltShape, stddev=0.03),
                                      name=name+'_W')
    
    bias = tf.Variable(tf.truncated_normal([numOutChannels]), name=name+'_b')

    # setup the convolutional layer operation
    outLayer = tf.nn.conv2d(inputData, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    outLayer += bias

    # apply a ReLU non-linear activation
    outLayer = tf.nn.relu(outLayer)

    # now perform max pooling
    ksize = [1, poolShape[0], poolShape[1], 1]
    
    strides = [1, poolShape[0], poolShape[1], 1]
    
    outLayer = tf.nn.max_pool(outLayer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return outLayer

# Create convolutional/pooling layers

layer1 = createConvLayer(x_image, 30, 120, [5, 5], [4, 4], name='layer1')

layer2 = createConvLayer(layer1, 120, 240, [5, 5], [4, 4], name='layer1' )


# Flatten output of convolutional/pooling layers into a vector

flattened = tf.reshape(layer2, [-1, 4 *4 *240 ])


# Create weights and bias variables for first dense layer, then activate with ReLU.

wd1 = tf.Variable(tf.truncated_normal([4 *4 *240, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')


denseLayer1 = tf.matmul(flattened, wd1) + bd1

denseLayer1 = tf.nn.relu(denseLayer1)

# Create weights and bias variables for second dense layer.

wd2 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd2')

beta = tf.Variable(tf.truncated_normal( [clinDataDim,2 ], stddev=0.01), name='beta')


denseLayer2 = tf.matmul(denseLayer1, wd2) + bd2 + tf.matmul(x_clin, beta )



# small is a small, positive value that is used to prevent NaN occurring
# by diving by zero or taking log of zero

small = tf.constant( 10**-10 , name="small")

  
# yhat is the prediction of the NN

yhat = tf.maximum( denseLayer2, small )



# Define a tf constant that is square root of 2.

root2 = tf.constant(  math.sqrt(2) , name="root2")

# These are clippped values that are used in the objective function.

Delta = tf.minimum( tf.math.abs( yhat[:, 0] - y ), 1000 )

sigma = tf.minimum( yhat[:, 1], 70 )


# Define objective function to be minimised.


objective = tf.reduce_mean( tf.divide( root2 * Delta, sigma ) + tf.log( root2 * sigma ) )


# This operation is the optimisation
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(objective)


# This operation initialises all the variables
initOp = tf.global_variables_initializer()



# Train the neural network.


with tf.Session() as sess:
   
    sess.run(initOp)
    
    totalBatches = math.ceil( n / batchSize )
    
    for epoch in range(epochs):
        
        for i in range(totalBatches):
            
            batch_y, batch_x_clin, batch_x_image = createBatch(batchSize)
            
            optim, objectiveValue = sess.run([optimiser, objective], 
   feed_dict={y: batch_y, x_clin: batch_x_clin, x_image: batch_x_image  })
           
        print( "Epoch", (epoch + 1), "average likelihood", "{:.3f}".format(-objectiveValue) )

    print("Training complete!")
    
    
    Yhat = np.empty([n, 2])
    Objective = 0
    
    for j in range(n):
        
        
        Y, X_clin, X_image = combineData(j)
        
        Y = np.array( [Y] )
        
        X_clin = np.array(  [  X_clin ]  )
        
        X_image = X_image[np.newaxis, : , :, : ]
        
        Yhat[j, :], obj = sess.run( [yhat, objective] ,  
            feed_dict={y: Y, x_clin: X_clin, x_image: X_image  }   )
        
        Objective += obj
        
    Objective = Objective / n
    
    print( 'Training set average likelihood', "{:.3f}".format(-Objective) )
   
#    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))









