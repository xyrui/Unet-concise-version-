# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 21:09:50 2018

@author: Administrator
"""
import tensorflow as tf
import numpy as np

def conv3x3(X, fil_num):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)
    ke_ini = tf.contrib.layers.variance_scaling_initializer()
    bi_ini = tf.constant_initializer(0)
    
    feature = tf.layers.conv2d(X, fil_num, 3, strides = 1, padding = 'same',
                                 activation = tf.nn.relu, kernel_regularizer = regularizer,
                                 kernel_initializer = ke_ini, bias_initializer = bi_ini )
    return feature

def up2x2(X, fil_num):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)
    ke_ini = tf.contrib.layers.variance_scaling_initializer()
    bi_ini = tf.constant_initializer(0)
    
    feature = tf.layers.conv2d_transpose(X, fil_num, 2, strides = 2, padding = 'same',
                                 activation = tf.nn.relu, kernel_regularizer = regularizer,
                                 kernel_initializer = ke_ini, bias_initializer = bi_ini )
    return feature

def conv1x1(X, fil_num):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)
    ke_ini = tf.contrib.layers.variance_scaling_initializer()
    bi_ini = tf.constant_initializer(0)
    
    feature = tf.layers.conv2d(X, fil_num, 1, strides = 1, padding = 'same',
                                 activation = tf.nn.relu, kernel_regularizer = regularizer,
                                 kernel_initializer = ke_ini, bias_initializer = bi_ini )
    return feature
    

def En_layer(name, X, fil_num):
    with tf.variable_scope(name):
        conv1 = conv3x3(X, fil_num)
        conv2 = conv3x3(conv1, fil_num)
        
        pool = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides = 2)
        
    return pool,conv2

def De_layer(name, X, E_X, fil_num):
    with tf.variable_scope(name):
        conv1 = up2x2(X, fil_num)
        merge = tf.concat([E_X, conv1], axis = 3)
        conv2 = conv3x3(merge, fil_num)
        
        return conv2
       

def Unet(input, out_channels):
    down1, fea1 = En_layer('en_layer_1', input, 64)
    down2, fea2 = En_layer('en_layer_2', down1, 128)
    down3, fea3 = En_layer('en_layer_3', down2, 256)
    down4, fea4 = En_layer('en_layer_4', down3, 512)
    _, fea5 = En_layer('en_layer_5', down4, 1024)
    
    up4 = De_layer('de_layer4', fea5, fea4, 512)
    up3 = De_layer('de_layer3', up4, fea3, 256)
    up2 = De_layer('de_layer2', up3, fea2, 128)
    up1 = De_layer('de_layer1', up2, fea1, 64)
    
    with tf.variable_scope('final'):
        output = conv1x1(up1, out_channels)
        
    return output
    
        
    
