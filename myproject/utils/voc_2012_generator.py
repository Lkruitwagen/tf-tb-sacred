import numpy as np
from skimage.io import imread         # read an image to a np array
from skimage.transform import resize  # resize an image
from skimage.util import crop, pad    # crop or pad an image

import tensorflow as tf

VOC_COLORMAP = {
    "background":    [0, 0, 0],
    "aeroplane":    [128, 0, 0],
    "bicycle":    [0, 128, 0],
    "bird":    [128, 128, 0],
    "boat":    [0, 0, 128],
    "bottle":    [128, 0, 128],
    "bus":    [0, 128, 128],
    "car":    [128, 128, 128],
    "cat":    [64, 0, 0],
    "chair":    [192, 0, 0],
    "cow":    [64, 128, 0],
    "diningtable":    [192, 128, 0],
    "dog":    [64, 0, 128],
    "horse":    [192, 0, 128],
    "motorbike":    [64, 128, 128],
    "person":    [192, 128, 128],
    "potted plant":    [0, 64, 0],
    "sheep":    [128, 64, 0],
    "sofa":    [0, 192, 0],
    "train":    [128, 192, 0],
    "tv/monitor":    [0, 64, 128],
}

### let's use our colormap to make a mask-generating function
def get_mask(image):
    # image: an 3d (WxHxRGBA) numpy array of the annotation image
    
    height, width = image.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP.keys())), dtype=np.float32)
    for label_index, (key, rgb_value) in enumerate(VOC_COLORMAP.items()):
        segmentation_mask[:, :, label_index] = np.all(image == rgb_value, axis=-1).astype(float)
    return segmentation_mask


def voc2012_generator(records, output_shape=(200,200), mode='random_crop'):
    ### a wrapper for our generator. Takes all our parameters and returns the generator.

    def _generator():
        ### The internal generator must not take any parameters.

        for r in records:

            # io
            x = (imread(r['image'])).astype(np.float32)  # <- again, don't normalise.
            ann = imread(r['annotation'])[:,:,0:3]       # drop the alpha channel
            y = get_mask(ann)                            # WHC, float32

            # reduce dimension of array
            if mode=='resize':
                x = resize(x,output_shape)
                y = resize(y,output_shape)
            elif mode=='random_crop':
                crop_width = [(0,0)]*3
                pad_width  = [(0,0)]*3
                for ax in [0,1]:
                    if x.shape[ax]>output_shape[ax]:
                        crop_val=np.random.choice(x.shape[ax]-output_shape[ax])
                        crop_width[ax] = (crop_val, x.shape[ax]-output_shape[ax]-crop_val)
                    elif x.shape[ax]<output_shape[ax]:
                        pad_val = np.random.choice(output_shape[ax]-x.shape[ax])
                        pad_width[ax] = (pad_val,output_shape[ax]- x.shape[ax]-pad_val)
                        
                x = crop(x, crop_width)
                x = pad(x, pad_width)
                y = crop(y, crop_width)
                y = pad(y, pad_width)

            yield tf.convert_to_tensor(x), tf.convert_to_tensor(y)
            
    return _generator

def trnvaltest_generators(records,shp,n_classes,trn_split, val_split, generator_mode, batch_size):
    
    generator_obj_trn = voc2012_generator(
        records[0:int(trn_split*len(records))], 
        output_shape=shp, 
        mode=generator_mode
    )
    generator_obj_val = voc2012_generator(
        records[int(trn_split*len(records)):int(val_split*len(records))], 
        output_shape=shp, 
        mode=generator_mode
    )
    generator_obj_test = voc2012_generator(
        records[int(val_split*len(records)):], 
        output_shape=shp, 
        mode=generator_mode
    )
    
    ds_voc_trn = (
        tf.data.Dataset.from_generator(
         generator_obj_trn,
         output_signature=(
             tf.TensorSpec(shape=(shp[0],shp[1],3), dtype=tf.float32),
             tf.TensorSpec(shape=(shp[0],shp[1],n_classes), dtype=tf.float32)))  # <- new shape!
        ) \
        .cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    ds_voc_val = (
        tf.data.Dataset.from_generator(
         generator_obj_val,
         output_signature=(
             tf.TensorSpec(shape=(shp[0],shp[1],3), dtype=tf.float32),
             tf.TensorSpec(shape=(shp[0],shp[1],n_classes), dtype=tf.float32)))  # <- new shape
        ) \
        .cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    ds_voc_test = (
        tf.data.Dataset.from_generator(
         generator_obj_test,
         output_signature=(
             tf.TensorSpec(shape=(shp[0],shp[1],3), dtype=tf.float32),
             tf.TensorSpec(shape=(shp[0],shp[1],n_classes), dtype=tf.float32)))  # <- new shape
        ) \
        .cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds_voc_trn, ds_voc_val, ds_voc_test