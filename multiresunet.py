"""
English :
    University of Guanajuato
    Science and Engineering Faculty
    Research group : DCI-Net
    Chief researcher : Dr. Carlos Padierna
    Student : Gustavo Magaña Lopez

        MultiResUNet 
        Implementation based on the following research paper :
        https://arxiv.org/abs/1902.04049

Español :
    Universidad de Guanajuato
    Division de Ciencias e Ingenierias
    Grupo de investigacion : DCI-Net
    Investigador responsable : Dr. Carlos Padierna
    Alumno : Gustavo Magaña Lopez

        MultiResUNet 
        Implementacion basada en el siguiente articulo :
        https://arxiv.org/abs/1902.04049
"""

##### Standard imports 
# Used for type-hints : 
from typing import List, Tuple

##### Machine Learning-Specific 
# Main API :
from tensorflow import keras as k
# Metrics :
from sklearn.metrics import jaccard_score as jaccard_index

##### Custom modules 
# Timing decorators :
from timing import time_this

def convolve(x, filters: int = 1, kernel_size: Tuple[int] = (3, 3), 
             padding: str ="same", strides: Tuple[int] = (1, 1), 
             activation: str = "relu", batch_norm: bool = True):
    """
        2D Convolutional layers with optional Batch Normalization
        Basically a wrapper for keras.layers.Conv2D, with some add-ons 
        for ease of use.

        Default values of keyword arguments are set to minimize verbosity
        when calling the function. Verify them to avoid specifing one with
        the same value as the default.

##### Arguments:
                x: Keras layer, the input to the feature map.
          filters: Integer representing the number of filters to use.
      kernel_size: Should probably be called shape.
                   Tuple with two integer values (number of rows, number of columns).
          padding: String that determines the padding mode.
                   'valid' or 'same'. See help(keras.layers.Conv2D)
          strides: Tuple of two integer values that represent the strides.
       activation: String that defines the activation function.
       batch_norm: Boolean flag, switches between using bias and BatchNormalization.
                   These two are complements so that :
                      batch_norm = True -->
                          BatchNormalization = True
                          use_bias = False

                      batch_norm = False -->
                          BatchNormalization = False
                          use_bias = True


##### Returns:
                x: A Keras layer.
    """

    use_bias = not batch_norm
    activations = list(
                    filter(
                      lambda x: x if x != 'serialize' and x != 'deserialize' else False, 
                      dir(k.activations)[10:] # These are the methods which are valid activations.
                    )
                  )  

    f = K.layers.Conv2D(
          filters=filters, 
          kernel_size=kernel_size, 
          strides=strides, 
          padding=padding, 
          use_bias=use_bias
        )

    y = f(x)
    if batch_norm:
        y = K.layers.BatchNormalization(scale=False)(y)

    if activation in activations:
        y = K.layers.Activation(activation)(y)

    return y
##

def MultiResBlock(prev_layer, U: int, alpha: float = 1.67, weights: List[float] = [0.167, 0.333, 0.5]):
    """

        As defined in the paper, with the possibility of 
        changing the weights of each one of the three successive
        convolutional layers.

        W is the number of filters of the convolutional layers 
        inside of the MultiResBlock, calculated from 'U' and 'alpha'
        as follows:

          W = alpha * U
      
##### Arguments:
          prev_layer: A Keras layer.
                   U: Integer value for the number of filters that would be used
                      in the analogue U-Net, to estimate W which is the number 
                      used in our model.
               alpha: Scaling constant, defaults to 1.67, which 
                      'keeps it comparable to the original
                       U-Net with slightly less parameters'.


             weights: A list containing float values, which should add up to one and 
                      in combination with W determine the number of filters in each one
                      of the successive convolutional layers inside the MultiResBlock.
                      Default values are taken from the article :
                       'Hence, we assign W/6, W/3, and W/2 filters to the three successive
                        convolutional layers respectively, as this combination achieved 
                        the best results in our experiments.'
            
 
##### Returns:
          out: A Keras layer.

    """
    
    W = alpha * U
    
    def_1x1 = {
      "filters": sum(map(lambda x: int(W * x), weights)), 
      "kernel_size": (1, 1), 
    }
    # 1x1 filter for conserving dimensions
    residual1x1 = convolve(prev_layer, **def_1x1)

    maps_kws = [
      dict(
        filters=int(W * i), 
        kernel_size=(3, 3), 
        activation="relu", 
        padding="same"
      ) for i in weights
    ]
    
    first  = convolve(prev_layers, **maps_kws[0])
    second = convolve(first, **maps_kws[1])
    third  = convolve(second, **maps_kws[2])

    # Concatenate successive 3x3 convolution maps :
    out = K.layers.Concatenate()([first, second, third])

    # And add the new 7x7 map with the 1x1 map, batch normalized
    out = K.layers.add([residual1x1, out])
    out = K.layers.Activation("relu")(out)

    return out
##


def ResPath(encoder_out, layers: int, n_filters: int, batch_norm: bool = True):
    """
    """

    def_1x1  = {
      "filters": n_filters,
      "kernel_size": (1, 1), 
    }
    
    def_3x3 = {
      "filters": n_filters,
      "kernel_size": (3, 3), 
    }

    # First block :
    x = convolve(encoder_out, **def_1x1) # Residual connection
    y = convolve(encoder_out, **def_3x3) # Feature map
    y = k.layers.Activation("relu")(
            k.layers.add(
                          [x, y]
            )
        )
    if batch_norm:
        y  = k.layers.BatchNormalization()(y)

    # Construct paths with variable length, adding feature maps.
    if layers is not None and layers > 1:
        for _ in range(layers - 1):
            x = y
            x = convolve(x, **def_1x1)
            y = convolve(y, **def_3x3)

            y = k.layers.add([x, y])
            y = k.layers.Activation("relu")(out)
            y = k.layers.BatchNormalization()(out)

    return y
##

def MultiResUNet(input_shape=(256, 256, 3)):
    """
    A Keras implementation of the MultiResUNet architecture as defined in the
    following paper:
        https://arxiv.org/abs/1902.04049
    
    This is a variant of the U-Net, with additional blocks and paths to help mitigate
    semantic gaps and to obtain better characteristics from the images and maps.
    
    Arguments:
        input_shape: Tuple of three integers (height, width, number of channels) that
                     describe the input images.
    
    Returns:
        model: A Keras model instance.
    """

    inputs = K.layers.Input((input_size))

    mresblock_1 = MultiResBlock(32, inputs)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_1)
    mresblock_1 = ResPath(32, mresblock_1, 4)

    mresblock_2 = MultiResBlock(64, pool_1)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_2)
    mresblock_2 = ResPath(64, mresblock_2, 3)

    mresblock_3 = MultiResBlock(128, pool_2)
    pool_3 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_3)
    mresblock_3 = ResPath(128, mresblock_3, 2)

    mresblock_4 = MultiResBlock(256, pool_3)
    pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_4)
    mresblock_4 = ResPath(256, mresblock_4)

    mresblock5 = MultiResBlock(512, pool_4)

    up_6 = K.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(
        mresblock5
    )
    up_6 = K.layers.Concatenate()([up_6, mresblock_4])
    mresblock_6 = MultiResBlock(256, up_6)

    up_7 = K.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(
        mresblock_6
    )
    up_7 = K.layers.Concatenate()([up_7, mresblock_3])
    mresblock7 = MultiResBlock(128, up_7)

    up_8 = K.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(
        mresblock7
    )
    up_8 = K.layers.Concatenate()([up_8, mresblock_2])
    mresblock8 = MultiResBlock(64, up_8)

    up_9 = K.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(
        mresblock8
    )
    up_9 = K.layers.Concatenate()([up_9, mresblock_1])
    mresblock9 = MultiResBlock(32, up_9)

    conv_10 = conv2d(mresblock9, 1, (1, 1), activation="sigmoid")

    model = K.models.Model(inputs=[inputs], outputs=[conv_10])

    return model
##

