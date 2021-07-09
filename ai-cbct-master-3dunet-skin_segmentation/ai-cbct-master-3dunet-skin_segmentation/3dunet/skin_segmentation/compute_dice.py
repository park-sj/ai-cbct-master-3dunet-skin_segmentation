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


from unet3d.losses import compute_per_channel_dice
def DiceCoefficient(input, target, epsilon=1e-6):
    return torch.mean(compute_per_channel_dice(input, target, epsilon=epsilon))


# class DiceCoefficient:
#     """Computes Dice Coefficient.
#     Generalized to multiple channels by computing per-channel Dice Score
#     (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
#     Input is expected to be probabilities instead of logits.
#     This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
#     DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
#     """
#
#     def __init__(self, epsilon=1e-6, **kwargs):
#         self.epsilon = epsilon
#
#     def __call__(self, input, target):
#         # Average across channels in order to get the final score
#         return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))
#
# def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
#     """
#     Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
#     Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
#
#     Args:
#          input (torch.Tensor): NxCxSpatial input tensor
#          target (torch.Tensor): NxCxSpatial target tensor
#          epsilon (float): prevents division by zero
#          weight (torch.Tensor): Cx1 tensor of weight per channel/class
#     """
#
#     # input and target shapes must match
#     assert input.size() == target.size(), "'input' and 'target' must have the same shape"
#
#     input = input.flatten()
#     target = target.flatten()
#     target = target.float()
#
#     # compute per channel Dice Coefficient
#     intersect = (input * target).sum(-1)
#
#     if weight is not None:
#         intersect = weight * intersect
#
#     # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
#     denominator = (input * input).sum(-1) + (target * target).sum(-1)
#     # denominator = (input).sum(-1) + (target).sum(-1)
#
#     return 2 * (intersect / denominator.clamp(min=epsilon))

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

    wb = xl.Workbook()
    sheet = wb.active
    sheet.title = 'DiceCoefficient'

    col_names = ['Hos_num', 'Style_dice', 'origin_dice']
    for seq, name in enumerate(col_names):
        sheet.cell(row=1, column=seq+1, value=name)

    style_full_Path = 'E:/0705_style_transfer/v1_Seg_True_Dcrop_True_2domains_seed_7777_seg_H_dcm'
    origin_full_Path = 'E:/0705_style_transfer/ct_seg_H'
    mask_full_Path = 'E:/0705_style_transfer/mask'

    style_full_Path = return_to_path_of_file(style_full_Path)
    origin_full_Path = return_to_path_of_file(origin_full_Path)
    mask_full_Path = return_to_path_of_file(mask_full_Path)

    assert len(style_full_Path) == len(origin_full_Path) == len(mask_full_Path), 'dcm and mask are not matched'

    for idx in range(len(style_full_Path)):
        stylePath = style_full_Path[idx]
        originPath = origin_full_Path[idx]
        maskPath = mask_full_Path[idx]

        style = _load_files(stylePath)
        origin = _load_files(originPath)
        mask = _load_files(maskPath)

        style = torch.sigmoid(style)
        style = (style > 0.5).float()

        dice_st = DiceCoefficient(input=style, target=mask).item()
        dice_or = DiceCoefficient(input=origin, target=mask).item()

        # dice_st = DiceCoefficient().__call__(input=style, target=mask).item()
        # dice_or = DiceCoefficient().__call__(input=origin, target=mask).item()

        print('style = ', dice_st)
        print('origin= ', dice_or)
        print('')

        sheet.append([originPath[originPath.rfind('/')+1:], dice_st, dice_or])

    wb.save('E:/0705_style_transfer/H_channel_dice_result.xlsx')