import tensorflow as tf

def UpBlock(n_filters, n_blocks):
    # make a function to build our upblocks flexible to the number of filters
    
    block_layers = []
    
    for _ in range(n_blocks):
        block_layers.append(tf.keras.layers.Activation('relu'))
        block_layers.append(tf.keras.layers.Conv2DTranspose(n_filters, 3, padding="same"))
        block_layers.append(tf.keras.layers.BatchNormalization())
        
    # and add the upsampling layer
    block_layers.append(tf.keras.layers.UpSampling2D(2))
    
    return block_layers

def make_model(shp, finetune_encoder=False):
    # make a U-Net like model for semantic segmentation
    
    vgg_encoder = tf.keras.applications.VGG16(
        include_top=False, 
        input_shape=(shp[0],shp[1],3),
        weights='imagenet', 
        pooling=None,         # <- in this case, we don't want any pooling on our final outputs
    )
    
    vgg_encoder.trainable = finetune_encoder
    
    # don't need to get the last maxpool layer
    encoder_output = vgg_encoder.get_layer('block5_conv3').output
    
    # get the featuremaps from the encoder so we can bridge them to the decoder
    block1_featuremap = vgg_encoder.get_layer('block1_conv2').output  # 224x224
    block2_featuremap = vgg_encoder.get_layer('block2_conv2').output  # 112x112
    
    # VGG doesn't have batch norm so let's do it ourselves:
    block1_featuremap = tf.keras.layers.Activation('relu')(block1_featuremap)
    block1_featuremap = tf.keras.layers.BatchNormalization()(block1_featuremap)
    block2_featuremap = tf.keras.layers.Activation('relu')(block2_featuremap)
    block2_featuremap = tf.keras.layers.BatchNormalization()(block2_featuremap)
    
    # the decoder with bridging:
    x = tf.keras.models.Sequential(UpBlock(128,2))(encoder_output)      # 28x28
    x = tf.keras.models.Sequential(UpBlock(64,2))(x)                   # 56x56
    x = tf.keras.models.Sequential(UpBlock(32,2))(x)                    # 112x112
    x = tf.keras.layers.Concatenate()([block2_featuremap,x])          # 112x112
    x = tf.keras.models.Sequential(UpBlock(32,2))(x)                    # 224x224
    x = tf.keras.layers.Concatenate()([block1_featuremap,x])          # 224x224
    
    # add some header layers:
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    output = tf.keras.layers.Conv2D(21, 1, padding="same", activation='softmax')(x)

    return tf.keras.models.Model(vgg_encoder.input, output)