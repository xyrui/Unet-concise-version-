# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:18:30 2018

@author: Administrator
"""
from model import Unet
import numpy as np
import argparse
import os
import tensorflow as tf

parser = argparse.ArgumentParser(description = 'Unet for denoising')
parser.add_argument('--mode', dest = 'mode', default = 'train', help = 'train or test')
parser.add_argument('--out_channel', dest = 'out_channel', default = 10, type = int, help = '#output channels')
parser.add_argument('--save_path', dest = 'save_path', default = './model', help = 'pretrained models are saved here')
args = parser.parse_args()

def train():
    X = np.float32(np.random.rand(2,400,240,10))
    
    input = tf.placeholder(tf.float32, shape = (None, None, None, 10))
    output = Unet(input, args.out_channel)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True, gpu_options = gpu_options)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        Y = sess.run(output, feed_dict = {input: X})
        print(Y.shape)
    
def test():
    pass
    
if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print('Er... Unrecognizable mode |-_-?|')
    

