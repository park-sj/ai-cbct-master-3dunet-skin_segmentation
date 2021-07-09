import time

import numpy as np
import torch
import SimpleITK as sitk
import os
import scipy.ndimage
import skimage.transform

from sklearn.cluster import MeanShift

from unet3d.utils import get_logger

logger = get_logger('UNet3DPredictor')


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)
        self.it = 0
        self.device = self.config['device']

    def predict(self):

        logger.info(f'Running prediction on {len(self.loader)} batches...')

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        
        # Run predictions on the entire input dataset
        with torch.no_grad():
            # for batch, indices in self.loader:
            for i, batch in enumerate(self.loader):

                # send batch to device
                batch = batch.to(self.device)

                # forward pass
                predictions = self.model(batch)
                predictions = predictions.cpu().numpy().astype(np.float32)
                
                # get io info
                output_file = os.path.join(self.config['loaders']['test']['file_paths'][0], 'save')
                input_dir = os.path.join(self.config['loaders']['test']['file_paths'][0], 'test')
                patient = os.listdir(input_dir)[i]
                
                if self.config['loaders']['dataset'] == 'PngDataset':
                    self._save_png(predictions, os.path.join(output_file, patient), os.path.join(input_dir, patient))
                else:
                    self._save_dicom(predictions, os.path.join(output_file, patient), os.path.join(input_dir, patient))

    
    @staticmethod
    def _save_png(newArray, filepath, template_dir):
        import glob
        import imageio   
        def _load_template(dir):
            imageList = glob.glob(dir + '/*.png')
            imageList = sorted(imageList, reverse=True) # 지금 테스트 해 본 데이터 대상으로는 reverse가 맞는데 프로메디우스 데이터에 해봐야한다.
            image = None
            for i in imageList:
                if image is None:
                    #image = imageio.imread(i)[:,:,0]
                    image = imageio.imread(i)
                    image = np.expand_dims(image, 0)
                else:
                    #image = np.concatenate((image, np.expand_dims(imageio.imread(i)[:,:,0], 0)), axis = 0)
                    image = np.concatenate((image, np.expand_dims(imageio.imread(i), 0)), axis=0)
            return image.shape
        
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        
        imgShape = _load_template(template_dir)
        logger.info("The template is loaded.")
        newArray = np.squeeze(newArray)
        newArray = skimage.transform.resize(newArray, imgShape, anti_aliasing=False)
        newArray[newArray > 0.5] = 255
        newArray[newArray <= 0.5] = 0
        newArray = newArray.astype(np.uint8)
        
        for i in range(imgShape[0]):
            image_slice = np.expand_dims(newArray[i,:,:],-1).repeat(3, axis=-1)
            imageio.imwrite(os.path.join(filepath,str(i).zfill(3) + '.png'),image_slice)

    @staticmethod
    def _save_dicom(newArray, filepath, template_dir):
        def _load_template(dir):
            logger.info("The template DCM directory is " + dir)
            assert os.path.isdir(dir), 'Cannot find the template directory'
            reader = sitk.ImageSeriesReader()
            dicomFiles = reader.GetGDCMSeriesFileNames(dir)
            reader.SetFileNames(dicomFiles)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            image = reader.Execute()
            img3d = sitk.GetArrayFromImage(image)
            return image, reader, img3d.shape

        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        oldImage, reader, imgShape = _load_template(template_dir)
        logger.info("The template is loaded.")
        newArray = np.squeeze(newArray)
        # newArray = scipy.ndimage.zoom(newArray, 2, order = 0)
        newArray = skimage.transform.resize(newArray, imgShape, anti_aliasing=False)
        # newArray = skimage.transform.rescale(newArray, 2, anti_aliasing = False)
        # newArray = skimage.transform.resize(newArray, (newArray.shape[0]*2, newArray.shape[1]*2, newArray.shape[2]*2), anti_aliasing=False)
        newArray[newArray > 0.5] = 1
        newArray[newArray <= 0.5] = 0
        # newArray *= 1000
        # for _ in range(10):
        #     newArray = scipy.ndimage.binary_erosion(scipy.ndimage.binary_dilation(newArray))
        #     newArray = scipy.ndimage.binary_dilation(scipy.ndimage.binary_erosion(newArray))
        # newArray = scipy.ndimage.zoom(newArray, 2, order = 1)
        # paddedArray = np.pad(newArray.astype(np.int16), ((4, 4),)*3, 'constant', constant_values =  ((0, 0),)*3)
        paddedArray = newArray.astype(np.int16)
        # paddedArray = np.flip()
        # paddedArray = paddedArray.transpose((2, 0, 1))
        # paddedArray = paddedArray.transpose((1,2,0))
        # assert paddedArray.shape == (600,)*3, 'You idiot messed up with output dimension. Check unet3d/predictor.py --p.'
        newImage = sitk.GetImageFromArray(paddedArray)

        newImage.CopyInformation(oldImage)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        sp_x, sp_y = reader.GetMetaData(0, "0028|0030").split('\\')
        # sp_z = reader.GetMetaData(0, "0018|0050")
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
                             ("0028|0100", "8"),
                             ("0028|0101", "8"),
                             ("0028|0102", "7"),
                             ("0028|0103", "1"),
                             ("0028|0002", "1"),
                             ("0008|0008", "DERIVED\\SECONDARY"),
                             ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                             ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]))))]
    #                         ("0008|103e", reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK")]
    #    print(series_tag_values)
        logger.info(f'Saving mask into {filepath}')
        tags_to_skip = ['0010|0010', '0028|0030', '7fe0|0010', '7fe0|0000', '0028|1052',
                        '0028|1053', '0028|1054', '0010|4000', '0008|1030', '0010|1001',
                        '0008|0080', '0010|0040']
        for i in range(newImage.GetDepth()):
            image_slice = newImage[:, :, i]
            # image_slice.CopyInformation(oldImage[:, :, i])
            for tag, value in series_tag_values:
                if (tag in tags_to_skip):
                    continue
                if i == 0:
                    try:
                        logger.info(f'{tag} | {value}')
                    except:
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
        logger.info(f'Saved mask into {filepath}')