"""
loss function definitions for deep abstaining classifier.
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pdb
import math

#for numerical stability
epsilon = 1e-7

#this might be changed from inside dac_sandbox.py (??)
total_epochs = 100
alpha_final = 1.0
alpha_init_factor = 64.

# loss calculation and alpha-auto-tune are rolled into one function.
# This is invoked after every iteration

class DACLoss(_Loss):

    def __init__(self, learn_epochs, device=None):
        super().__init__()
        self.learn_epochs = learn_epochs

        #self.alpha = alpha
        # if self.use_cuda:
        #   self.alpha_var =  Variable(torch.Tensor([self.alpha])).to(self.device)
        # else:
        #   self.alpha_var =  Variable(torch.Tensor([self.alpha]))
        self.alpha_var = None
        #self.alpha_var = 1.0

        # exponentially weighted moving average for alpha_thresh
        self.alpha_thresh_ewma = None
        # mu parameter for EWMA;
        self.ewma_mu = 0.05
        # for alpha initiliazation
        self.curr_alpha_factor  = None
        # linear increase factor of alpha during abstention phase
        self.alpha_inc = None
        self.alpha_set_epoch = None

        self.eps_p_out = torch.Tensor([1. - epsilon]).to(device)

    def __call__(self, input_batch, target_batch, epoch):

        def obtain_abstain_prob():
            # calculate cross entropy only over true classes
            h_c = F.cross_entropy(input_batch[:, :-1], target_batch, reduction='none')
            #p_out = F.softmax(F.log_softmax(input_batch, dim=1), dim=1)
            p_out = F.softmax(input_batch, dim=1)
            # probabilities of abstention  class
            p_out_abstain = p_out[:, -1]
            # avoid numerical instability by upper-bounding p_out_abstain to never be more than
            # 1 - eps since we have to take log(1 - p_out_abstain) later.
            p_out_abstain = torch.min(p_out_abstain, self.eps_p_out)
            #update alpha_thresh_ewma
            if self.alpha_thresh_ewma is None:
                self.alpha_thresh_ewma = ((1. - p_out_abstain) * h_c).mean()
            else:
                self.alpha_thresh_ewma = self.ewma_mu * ((1. - p_out_abstain) * h_c).mean() + \
                                         (1. - self.ewma_mu) * self.alpha_thresh_ewma
            return h_c, p_out_abstain

        # begin call main
        if epoch <= self.learn_epochs:
            h_c, _ = obtain_abstain_prob()
            loss = F.cross_entropy(input_batch, target_batch, reduction='none')
            # print("\nloss details (pre abstention): %d,%f,%f,%f,%f\n" %(epoch,p_out_abstain.mean(),loss.mean(),h_c.mean(),
            #   self.alpha_thresh_ewma))
            return loss.mean()
        else:
            h_c, p_out_abstain = obtain_abstain_prob()
            if self.alpha_var is None:
                # hasn't been initialized. do it now
                # we create a freshVariable here so that the history of alpha_var
                # computation (which depends on alpha_thresh_ewma) is forgotten. This
                # makes self.alpha_var a leaf variable, which will not be differentiated.
                self.alpha_var = Variable(self.alpha_thresh_ewma / alpha_init_factor)
                self.alpha_inc = (alpha_final - self.alpha_var) / (total_epochs - epoch)
                self.alpha_set_epoch = epoch
            else:
                # we only update alpha every epoch
                if epoch > self.alpha_set_epoch:
                    self.alpha_var = Variable(self.alpha_var + self.alpha_inc)
                    self.alpha_set_epoch = epoch

            loss = (1. - p_out_abstain) * h_c - self.alpha_var * torch.log(1. - p_out_abstain)
            #calculate entropy of the posterior over the true classes.
            #h_p_true = -(F.softmax(input_batch[:, :-1], dim=1) * F.log_softmax(input_batch[:,:-1],dim=1)).sum(1)
            #loss = loss - self.kappa*h_p_true
            # print("\nloss details (during abstention): %d, %f,%f,%f,%f\n" %(epoch,p_out_abstain.mean(), h_c.mean(),
            #       self.alpha_thresh_ewma, self.alpha_var))
            return loss.mean()

