# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:44:52 2021

@author: user
"""

import SimpleITK as sitk
import imageio
import glob
import sys
import time
import os
import numpy as np
import skimage.transform

def loadPng(dir):
    assert os.path.isdir(dir), f'Cannot find the directory {dir}'
    imageList = glob.glob(dir + '/*.png')
    imageList = sorted(imageList, reverse=True) # 지금 테스트 해 본 데이터 대상으로는 reverse가 맞는데 프로메디우스 데이터에 해봐야한다.
    image = None
    for i in imageList:
        if image is None:
            image = imageio.imread(i)[:,:,0]
            image = np.expand_dims(image, 0)
        else:
            image = np.concatenate((image, np.expand_dims(imageio.imread(i)[:,:,0], 0)), axis = 0)
    return image

def loadDcm(dir):
    assert os.path.isdir(dir), f'Cannot find the directory {dir}'
    reader = sitk.ImageSeriesReader()
    dicomFiles = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicomFiles)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    img3d = sitk.GetArrayFromImage(image)
    return img3d.shape, image, reader

def saveDcm(img3d, oldImage, reader, filepath):
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
        
    img3d = img3d.astype(np.int16)
    newImage = sitk.GetImageFromArray(img3d)

    newImage.CopyInformation(oldImage)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    sp_x, sp_y = reader.GetMetaData(0, "0028|0030").split('\\')
    _, _, z_0 = reader.GetMetaData(0, "0020|0032").split('\\')
    _, _, z_1 = reader.GetMetaData(1, "0020|0032").split('\\')
    spacing_ratio = np.array([1, 1, 1], dtype=np.float64)
    sp_z = abs(float(z_0) - float(z_1))
    sp_z = float(sp_z) / spacing_ratio[0]
    sp_x = float(sp_x) / spacing_ratio[1]
    sp_y = float(sp_y) / spacing_ratio[2]
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = newImage.GetDirection()
    series_tag_values = [(k, reader.GetMetaData(0, k)) for k in reader.GetMetaDataKeys(0)] + \
                         [("0008|0031", modification_time),
                         ("0008|0021", modification_date),
                         ("0028|0100", "16"),
                         ("0028|0101", "16"),
                         ("0028|0102", "15"),
                         ("0028|0103", "1"),
                         ("0028|0002", "1"),
                         ("0008|0008", "DERIVED\\SECONDARY"),
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]))))]
    tags_to_skip = ['0010|0010', '0028|0030', '7fe0|0010', '7fe0|0000', '0028|1052',
                    '0028|1053', '0028|1054', '0010|4000', '0008|1030', '0010|1001',
                    '0008|0080', '0010|0040']
    for i in range(newImage.GetDepth()):
        image_slice = newImage[:, :, i]
        # image_slice.CopyInformation(oldImage[:, :, i])
        for tag, value in series_tag_values:
            if (tag in tags_to_skip):
                continue
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        image_slice.SetMetaData('0020|0032', reader.GetMetaData(i, "0020|0032"))
        image_slice.SetMetaData("0020|0013", str(i))
        image_slice.SetMetaData('0028|0030', '\\'.join(map(str, [sp_x, sp_y])))
        image_slice.SetSpacing([sp_x, sp_y])
        image_slice.SetMetaData("0018|0050", str(sp_z))
        writer.SetFileName(os.path.join(filepath, str(i).zfill(3) + '.dcm'))
        writer.Execute(image_slice)


def png2dcm(pngPath, dcmLoadPath, dcmSavePath):
    pngImage = loadPng(pngPath)
    targetShape, sitkImage, sitkReader = loadDcm(dcmLoadPath)
    pngImage = pngImage.astype(np.float64)
    pngImage = pngImage/255 * 2000 - 750
    pngImage = skimage.transform.resize(pngImage, targetShape)
    saveDcm(pngImage, sitkImage, sitkReader, dcmSavePath)
    
if __name__ == '__main__':
    pngPath = 'C:/Users/user/Desktop/ai-cbct-master-3dunet-skin_segmentation/ai-cbct-master-3dunet-skin_segmentation/3dunet/skin_segmentation/resources/io/save/1'
    dcmLoadPath = 'E:/0705_style_transfer/ct/DD/DD_1'
    dcmSavePath = 'C:/Users/user/Desktop/ai-cbct-master-3dunet-skin_segmentation/ai-cbct-master-3dunet-skin_segmentation/3dunet/skin_segmentation/resources/io/save/dcm'

    # if len(sys.argv) != 4:
    #     pngPath = 'E:/2021_07_05_segmentation result/JW/100_seg'
    #     dcmLoadPath = 'E:/2021_07_05_segmentation result/JW/JW_100'
    #     dcmSavePath = 'E:/2021_07_05_segmentation result/JW/100_seg_dcm'
    # else:
    #     pngPath = sys.argv[1]
    #     dcmLoadPath = sys.argv[2]
    #     dcmSavePath = sys.argv[3]
    
    png2dcm(pngPath, dcmLoadPath, dcmSavePath)