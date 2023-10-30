from torch import nn 
import torch
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()

    
    def forward(self,input,target):

        #input = F.sigmoid(input)

        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()

        dice = 1 - (2 * intersection + 1.0)/(input.sum() + target.sum() + 1.0)
        #BCE = F.binary_cross_entropy(input,target,reduction="mean")
        #DICE_BCE = BCE + dice
        
        return dice
        
