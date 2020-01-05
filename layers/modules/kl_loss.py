import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class KLLoss(nn.Module):
# def KLLoss(xg,xe,alpha):
    """
    Kl-loss function for bounding box regression from CVPR 2019 paper:
    Bounding Box Regression with Uncertainty for Accurate Object Detection
    by Yihui He, Chenchen Zhu, Jianren Wang. Marios Savvides, Xiangyu Zhang

    It is a replacement for the Smooth L1 loss often used in bounding box regression.

    The regression loss for a coordinate depends on |xg − xe| > 1 or not:

    Loss |xg − xe| ≤ 1:

        Lreg1 ∝ e^{−α} * 1/2(xg − xe)^2 + 1/2α

    and if |xg − xe| > 1, Loss:

        Lreg2 = e^{−α} (|xg − xe| − 1/2) + 1/2α

    PyTorch implementation by Jasper Bakker (JappaB @github)
    """
    def __init__(self, loc_loss_weight=1.0):
        super(KLLoss, self).__init__()

        # Insert your own parameters here if you want to adjust the KL-Loss function

        # option to adjust the size of the loss
        self.loc_loss_weight = loc_loss_weight

    def forward(self,xg,xe,alpha):

        """
        :param xg: The ground truth of the bounding box coordinates in x1y1x2y2 format
            shape: [number_of_boxes, 4]
        :param xe: The estimated bounding box coordinates in x1y1x2y2 format
            shape: [number_of_boxes, 4]
        :param alpha: The log(sigma^2) of the bounding box coordinates in x1y1x2y2 format
            shape: [number_of_boxes, 4]
        :return: total_kl_loss
        """

        assert (xg.shape == xe.shape and xg.shape == alpha.shape),"The shapes of the input tensors must be the same"


        smooth_l1 = F.smooth_l1_loss(xe,xg, reduction='none')

        # e^{-α}
        exp_min_alpha = torch.exp(-alpha)

        # 1/2α
        half_alpha = 0.5*alpha

        total_kl_loss = (exp_min_alpha * smooth_l1 + half_alpha).sum()
        # total_kl_loss = total_kl_loss.sum()

        #
        # # xg − xe
        # delta = xg-xe
        #
        # # |xg − xe|
        # abs_delta = torch.abs(delta)
        #
        # ## mask for Lreg1 and Lreg2
        # Lreg1_mask = abs_delta.le(1.0) # |xg − xe| ≤ 1
        # Lreg2_mask = abs_delta.gt(1.0) # |xg − xe| > 1
        #
        # ## calculate all elements for Lreg1
        # # (xg − xe) for Lreg1
        # delta_Lreg1 = delta[Lreg1_mask]
        #
        # # e^{-α}
        # exp_min_alpha1 = torch.exp(-alpha[Lreg1_mask])
        #
        # # 1/2α
        # half_alpha1 = 0.5*alpha[Lreg1_mask]
        #
        # L_reg1 = exp_min_alpha1 * 0.5 * torch.pow(delta_Lreg1,2) + half_alpha1
        # L_reg1 = L_reg1.sum()
        #
        # ## calculate all elements for Lreg2
        # # |xg − xe| for Lreg2
        #
        #
        #
        # abs_delta_Lreg2 = abs_delta[Lreg2_mask]
        #
        # # e^{-α}
        # exp_min_alpha2 = torch.exp(-alpha[Lreg2_mask])
        #
        # # 1/2α
        # half_alpha2 = 0.5*alpha[Lreg2_mask]
        #
        # L_reg2 = exp_min_alpha2 * (abs_delta_Lreg2 - 0.5) + half_alpha2
        # L_reg2 = L_reg2.sum()
        #
        #
        # ## total
        # total_kl_loss = L_reg1+L_reg2
        # # total_kl_loss *= self.loc_loss_weight

        # todo: remove after debugging
        # print()
        # print('Debug kl-loss: ')
        # print('delta', delta)
        # print('abs_delta', abs_delta)
        # print('alpha', alpha)
        # print('exp_min_alpha1', exp_min_alpha1)
        # print('exp_min_alpha1', exp_min_alpha2)
        # print('Lreg1mask', Lreg1_mask.sum())
        # print('Lreg2mask', Lreg2_mask.sum())

        return total_kl_loss


