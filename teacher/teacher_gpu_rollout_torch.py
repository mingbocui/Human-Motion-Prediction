from __future__ import division
import os 
import numpy as np
import torch
import random
np.random.seed(1234)

# from utils import *
from policy_adaptive_torch import * 

class TeacherGPURollout(object):
    """docstring for TeacherGPURollout"""
    def __init__(self, args):
        self.forward_steps = args.rollout_steps
        self.step_size = args.rollout_rate
        self.method = args.rollout_method
        self.args = args
        self.policy = PolicyAdaptive(self.step_size, self.method)
        self.log = False

    def set_env(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator

    # def sigmoid_cross_entropy_with_logits(self, x, y):
    #     try:
    #         return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    #     except:
    #         return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    # def compute_real_sigmoid(self, real_batch):
    #     try:
    #         real_sigmoid, real_logits = self.gan.discriminator(real_batch, is_training=False, reuse=True)
    #     except:
    #         real_sigmoid, real_logits = self.gan.discriminator(real_batch, is_reuse=True)
    #     real_sigmoid =  tf.reduce_mean(real_sigmoid)
    #     return real_sigmoid

    def bce_loss(self, input, target):
        """
        Numerically stable version of the binary cross-entropy loss function.
        As per https://github.com/pytorch/pytorch/issues/751
        See the TensorFlow docs for a derivation of this formula:
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        Input:
        - input: PyTorch Tensor of shape (N, ) giving scores.
        - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

        Output:
        - A PyTorch Tensor containing the mean BCE loss over the minibatch of
          input data.
        """
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

    def gan_d_loss_real(self, scores_real):
        """
        Input:
        - scores_real: Tensor of shape (N,) giving scores for real samples

        Output:
        - loss: Tensor of shape (,) giving GAN discriminator loss for real samples
        """
        y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
        loss_real = self.bce_loss(scores_real, y_real)
        return loss_real

    def gan_d_loss_fake(self, scores_fake):
        """
        Input:
        - scores_fake: Tensor of shape (N,) giving scores for fake samples

        Output:
        - loss: Tensor of shape (,) giving GAN discriminator loss for fake samples
        """
        y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
        loss_fake = self.bce_loss(scores_fake, y_fake)
        return loss_fake

    def compute_real_sigmoid(self, traj_real_rel, seq_start_end):
        scores_real = self.discriminator(None, traj_real_rel, seq_start_end)
        return self.gan_d_loss_real(scores_real)

    # def compute_forward_sigmoid_and_grad(self, forward_batch):
    #     try: 
    #         forward_sigmoid, forward_logits = self.gan.discriminator(forward_batch, reuse=True)
    #     except: 
    #         forward_sigmoid, forward_logits = self.gan.discriminator(forward_batch, is_reuse=True)
    #     g_forward_loss = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(forward_logits, tf.ones_like(forward_sigmoid)))
    #     forward_grad = tf.gradients(g_forward_loss, forward_batch)[0]
    #     return forward_sigmoid, forward_grad
    
    def compute_forward_sigmoid_and_grad(self, forward_batch, seq_start_end):
        with torch.enable_grad():
            forward_batch.requires_grad_()
            forward_logits = self.discriminator(None, forward_batch, seq_start_end)
            forward_sigmoid = torch.sigmoid(forward_logits)
            g_forward_loss = self.gan_d_loss_real(forward_logits)
            forward_grad = torch.autograd.grad(g_forward_loss, forward_batch)
        return forward_sigmoid, forward_grad


    def build_teacher(self, fake_batch, real_batch, seq_start_end):
        task = 'traj'
        if task == '2DGaussian':
            pass

        else:
            with torch.no_grad():
                len_seq = fake_batch.size(0)
                batch = fake_batch.size(1)
                coord = fake_batch.size(2)
                self.real_batch = real_batch.clone()
                self.first_batch = fake_batch.clone()
                self.forward_batch = fake_batch.clone()
                self.optimal_batch = fake_batch.clone()
                self.optimal_step_init = torch.ones([1, self.args.batch_size]).cuda()
                self.init_fake_sigmoid, self.forward_grad = self.compute_forward_sigmoid_and_grad(self.forward_batch, seq_start_end)
                

                # real_batch = tf.Variable(tf.constant(self.data[:self.args.batch_size], dtype=tf.float32), trainable=False)
                # forward_batch = tf.Variable(tf.constant(fake_batch), trainable=False)
                # forward_sigmoid = tf.Variable(tf.constant(fake_sigmoid), trainable=False)
                # forward_grad = tf.Variable(tf.constant(fake_grad), trainable=False)

                # optimal_batch = tf.Variable(tf.constant(fake_batch), trainable=False)
                # optimal_step = tf.Variable(tf.zeros([1, self.args.batch_size]), trainable=False)
                # self.sess.run(tf.variables_initializer([real_batch, forward_batch, forward_sigmoid, forward_grad, optimal_batch, optimal_step]))

                self.real_sigmoid = self.compute_real_sigmoid(self.real_batch, seq_start_end)
                self.default_sigmoid = torch.squeeze(self.init_fake_sigmoid)
                self.forward_loss = self.real_sigmoid - torch.squeeze(self.init_fake_sigmoid)
                # optimal search 
                self.optimal_sigmoid = torch.squeeze(self.init_fake_sigmoid)            
                self.optimal_loss = self.real_sigmoid - torch.squeeze(self.init_fake_sigmoid)            
                self.optimal_step = torch.squeeze(self.optimal_step_init)
                # forward_batch = self.forward_batch
                # forward_grad = self.forward_grad
                # optimal_batch = self.optimal_batch
                # recursive forward search 
                for i in range(self.forward_steps):

                    # forward update 
                    self.forward_batch = self.policy.apply_gradient(self.forward_batch, self.forward_grad, self.forward_loss)
                    
                    # clip to image value range 
                    # Clip trajectory 
                    self.forward_batch = torch.cat((self.real_batch[:8],self.forward_batch[8:]))
                    # self.forward_batch = torch.clamp(self.forward_batch, clip_value_min=-1.0, clip_value_max=1.0)

                    # tf.print(self.sess.run(self.forward_batch)[:3,0,0,0])
                    # self.forward_batch = self.sgd_step(self.forward_batch, self.forward_grad, self.forward_loss)
                    # # Add print operation
                    # forward_batch = tf.Print(forward_batch, [forward_batch], message="This is a: ")

                    # compute current value and next grad
                    self.forward_sigmoid, self.forward_grad = self.compute_forward_sigmoid_and_grad(self.forward_batch, seq_start_end)
                
                    # states
                    self.forward_loss = self.real_sigmoid - torch.squeeze(self.forward_sigmoid)

                    # comparison
                    self.indices_update = torch.le(self.forward_loss, self.optimal_loss)
                    # indices_update = (optimal_loss - forward_loss) > 0
                    self.optimal_loss = torch.where(self.indices_update, self.forward_loss, self.optimal_loss)
                    self.optimal_sigmoid = torch.where(self.indices_update, torch.squeeze(self.forward_sigmoid), self.optimal_sigmoid)
                    # optimal_loss[indices_update] = forward_loss[indices_update]
                   
                    self.forward_batch = self.forward_batch.permute(1, 0, 2)
                    # self.forward_batch = self.forward_batch.contiguous().view(batch, -1)
                    self.optimal_batch = self.optimal_batch.permute(1, 0, 2)
                    # self.optimal_batch = self.optimal_batch.contiguous().view(batch, -1)

                    # print(self.indices_update.shape)
                    # print(self.forward_batch.shape)
                    # print(self.optimal_batch.shape)
                    self.optimal_batch = torch.where(self.indices_update.unsqueeze(dim=1).unsqueeze(dim=1), self.forward_batch, self.optimal_batch)
                    # self.forward_batch = self.forward_batch.view(batch, len_seq, coord)
                    self.forward_batch = self.forward_batch.permute(1, 0, 2)
                    # self.optimal_batch = self.optimal_batch.view(batch, len_seq, coord)
                    self.optimal_batch = self.optimal_batch.permute(1, 0, 2)
                    # print(self.optimal_batch.shape)
                    # optimal_batch[indices_update,:,:,:] = forward_batch[indices_update,:,:,:]
                    self.optimal_step = torch.where(self.indices_update, (i+1)*torch.ones_like(self.optimal_step), self.optimal_step)
                    # optimal_step[indices_update] = i+1

                # reset teacher
                self.policy.reset_moving_average()

                self.optimal_grad = (self.first_batch - self.optimal_batch)
                self.optimal_step_fin = torch.clamp(self.optimal_step, min=1.0, max=10000.0)
                self.optimal_grad = self.optimal_grad / (torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.optimal_step_fin, dim=1), dim=2), dim=3))

                # self.optimal_batch = tf.clip_by_value(self.optimal_batch, clip_value_min=-1.0, clip_value_max=1.0)

                return self.optimal_grad, self.optimal_batch
            # self.optimal_batch = optimal_batch
            # TODO: precision 32 vs 64
            # assert np.max(np.abs(fake_grad - optimal_grad)) < 1e-8

