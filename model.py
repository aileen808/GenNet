# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np

from datasets import *
from ops import *


class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.build_model(flags)

    def descriptor(self, inputs, is_training=True, reuse=False):
        ####################################################
        # Define network structure for descriptor.
        # Recommended structure:
        # conv1: channel 64 kernel 4*4 stride 2
        # conv2: channel 128 kernel 2*2 stride 1
        # fc: channel output 1
        # conv1 - bn - relu - conv2 - bn - relu -fc
        ####################################################
        with tf.variable_scope("des", reuse=reuse):
            
            bs = tf.shape(inputs)[0]
            net = leaky_relu(batch_norm(conv2d(inputs, output_dim = 64, kernal = (4,4), strides = (2,2), name='d_conv1'), \
                train=is_training, name='d_bn1'))
            net = leaky_relu(batch_norm(conv2d(inputs, output_dim = 128, kernal = (2,2), strides = (1,1), name='d_conv2'), \
                train=is_training, name='d_bn2'))
            net = tf.reshape(net, [bs, 64*64*128])
            
            out = linear(net, 1, scope='d_fc3')

            return out


    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        def cond(i, samples):
            return tf.less(i, flags.T)

        def body(i, samples):
            noise = tf.random_normal(shape=tf.shape(samples), name='noise')
            syn_res = self.descriptor(samples, is_training=True, reuse=True)
            grad = tf.gradients(syn_res, samples, name='grad_des')[0]
            samples = samples - 0.5 * flags.delta**2 * (samples / flags.ref_sig**2 - grad)
            samples = samples + flags.delta * noise
            return tf.add(i, 1), samples

        i = tf.constant(0)
        i, samples = tf.while_loop(cond, body, [i, samples])

        return samples

    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        self.original_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])
        self.synthesized_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])
        self.m_original = self.descriptor(self.original_images)
        self.m_synthesized = self.descriptor(self.synthesized_images, reuse=True)
        self.train_loss = tf.subtract(tf.reduce_mean(
                self.m_synthesized), tf.reduce_mean(self.m_original))
        #self.train_loss = tf.reduce_sum(tf.squared_difference(self.m_synthesized, self.m_original))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'des' in var.name]

        self.train_op = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.train_loss, var_list=self.d_vars)
        self.sampling_op = self.Langevin_sampling(self.synthesized_images, flags)

    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)

        saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


        tf.summary.scalar('loss',self.train_loss)
        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

        print(" Start training ...")

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################

        #Get the mean of real image as starting point of Langevin dynamics
        train_data_mean = np.copy(train_data)
        for i in range(train_data.shape[0]):
            for channel in range(3):
                train_data_mean[i,:,:,channel] = np.mean(train_data[i,:,:,channel])
        #Save the mean image at the very beginning 
       
        save_images(train_data_mean, self.sample_dir + '/mean_data.png' )
        #save the orignal images as reference
        save_images(train_data, self.sample_dir + '/real_data.png' )

        


        for epoch in np.arange(flags.epoch):
            syn_img = self.sess.run(self.sampling_op, feed_dict = {self.synthesized_images: train_data_mean})
            _, summary_loss, loss = self.sess.run([self.train_op, summary_op, self.train_loss], \
                feed_dict={self.original_images: train_data, self.synthesized_images: syn_img})

            if np.mod(epoch,20) == 0:
                save_images(syn_img, self.sample_dir + '/test_train_{:02d}.png'.format(epoch)) 
                self.writer.add_summary(summary_loss, global_step=epoch) 

            print("Epoch: [%2d]  g_loss: %.8f" % (epoch, loss))