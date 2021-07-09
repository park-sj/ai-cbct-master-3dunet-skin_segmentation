import torch
import SimpleITK as sitk
from os import listdir
import os
from unet3d.utils import expand_as_one_hot
import openpyxl as xl
import csv
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)

class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = input.flatten()
    target = target.flatten()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    # denominator = (input * input).sum(-1) + (target * target).sum(-1)
    denominator = (input).sum(-1) + (target).sum(-1)

    return 2 * (intersect / denominator.clamp(min=epsilon))

# def dice_coef(input, target):
#     if input.shape != target.shape:
#         raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
#     else:
#         intersection = np.logical_and(input, target)
#         value = (2. * intersection.sum())  / (input.sum() + target.sum())
#     return value
#
# def dice_coef_2(y_true, y_pred, epsilon=1e-6):
#     y_true_f = (y_true).flatten()
#     y_pred_f = (y_pred).flatten()
#     intersection = (y_true_f * y_pred_f).sum()
#     return (2. * intersection + epsilon) / ((y_true_f).sum() + (y_pred_f).sum() + epsilon)

def return_to_path_of_file(path):

    ids = []

    for dir in listdir(path):
        for file in listdir(path + '/' + dir):
            ids.append(path + '/' + dir + '/' + file)

    return ids

def _load_files(dir):
    assert os.path.isdir(dir), 'Cannot find the dataset directory'
    # logger.info(f'Loading data from {dir}')
    reader = sitk.ImageSeriesReader()
    dicomFiles = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicomFiles)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    img3d = sitk.GetArrayFromImage(image)
    # img3d = img3d.transpose((1,2,0))
    return torch.from_numpy(img3d).type(torch.FloatTensor)

if __name__ == '__main__':

    # stylePath = 'E:/2021_07_05_segmentation result/DD/1_seg_dcm'
    # stylePath = 'E:/0705_style_transfer/v1_Seg_True_Dcrop_True_2domains_seed_7777_seg_dcm/DD/1'
    stylePath = 'C:/Users/user/Desktop/ai-cbct-master-3dunet-skin_segmentation/ai-cbct-master-3dunet-skin_segmentation/3dunet/skin_segmentation/resources/io/save/dcm'

    originPath = 'E:/0705_style_transfer/ct_seg/DD/DD_1'
    maskPath = 'E:/0705_style_transfer/mask/DD/DD_1'

    style = _load_files(stylePath)
    origin = _load_files(originPath)
    mask = _load_files(maskPath)

    style = torch.sigmoid(style)
    style = (style > 0.5).float()

    # style[:, 0:60, :] = 0.0

    # print(torch.eq(style, mask).sum())
    # print(torch.eq(origin, mask).sum())
    # exit()

    # print(style.shape)
    #
    # fl_st = style.flatten()
    # fl_mask = mask.flatten()
    #
    # result = fl_st() * fl_mask
    # print(result.sum())
    # exit()

    # print('style = ', dice_coef(input=style, target=mask))
    # print('origin= ', dice_coef(input=origin, target=mask))

    # print('style = ', dice_coef_2(y_pred=style, y_true=mask))
    # print('origin= ', dice_coef_2(y_pred=origin, y_true=mask))
    # exit()

    print('style = ', DiceCoefficient().__call__(input=style, target=mask))
    print('origin= ', DiceCoefficient().__call__(input=origin, target=mask))
