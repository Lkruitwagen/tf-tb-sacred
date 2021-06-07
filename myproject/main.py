import os


from myproject.utils import voc_2012_generator
from myproject.models import mymodel
from myproject.train import train
from myproject import ex

def vgg16_premapper(_x, _y):                     # sample and target are now tf tensors
    return tf.keras.applications.vgg16.preprocess_input(_x), _y      # return the (image, label) tuple

@ex.automain
def main(_run, _log, data_root, input_shape, finetune_encoder, trn_split, val_split, generator_mode, batch_size, n_epochs):
    # _run is our sacred experiment run, _log is our sacred logger
    
    ### get data records
    # read in unique idxs from 'trainval.txt'. We'll split training and validation later.
    with open(os.path.join(data_root,'VOCdevkit','VOC2012','ImageSets','Segmentation','trainval.txt'),'r') as f:
        idxs = [line.strip() for line in f.readlines()]  # strip any line breaks, returns, white space
        
    # make a records list of dicts
    records = [
        {
            'image':os.path.join(data_root,'VOCdevkit','VOC2012','JPEGImages',idx+'.jpg'),
            'annotation':os.path.join(data_root,'VOCdevkit','VOC2012','SegmentationClass',idx+'.png'),
        }
        for idx in idxs
    ]
    
    ### set up our generators
    ds_voc_trn, ds_voc_val, ds_voc_test = voc_2012_generator.trnvaltest_generators(records,input_shape,n_classes,trn_split, val_split, generator_mode, batch_size)
    
    ds_voc_trn = ds_voc_trn.map(vgg16_premapper, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    ds_voc_val = ds_voc_val.map(vgg16_premapper, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    ds_voc_test = ds_voc_test.map(vgg16_premapper, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    
    ### initialise our model, optimizer, and loss_fn
    model = mymodel.make_model(input_shape, finetune_encoder)
    
    optimizer=tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    ### set up out tensorboard writer
    writer = SummaryWriter(os.path.join(os.getcwd(), 'experiments', 'tensorboard',_run._id))
    
    ### call our training loop
    pbar_len = int(len(records)*val_split)//batch_size + 1
    
    train(_log, pbar_len, n_epochs, model, optimizer, ds_voc_trn, ds_voc_val, writer)
    
    ### any postprocessing
    # pass
    
    ### save our model
    
    ### add it as an Sacred artefact
    