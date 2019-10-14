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
             padding="same", strides=(1, 1), activation="relu"):
    """
        2D Convolutional layers with optional Batch Normalization
        Basically a wrapper for keras.layers.Conv2D, with some add-ons 
        for ease of use.

##### Arguments:
                x: Keras layer, the input to the feature map.
          filters: Integer representing the number of filters to use.
      kernel_size: Should probably be called shape.
                   Tuple with two integer values (number of rows, number of columns).
          padding: String that determines the padding mode.
                   'valid' or 'same'. See help(keras.layers.Conv2D)
          strides: Tuple of two integer values that represent the strides.
       activation: String that defines the activation function.

##### Returns:
                x: A Keras layer.
    """

    x = K.layers.Conv2D(
        filters, shape, strides=strides, padding=padding, use_bias=False
    )(x)
    x = K.layers.BatchNormalization(scale=False)(x)

    if activation is None:
        return x

    x = K.layers.Activation(activation)(x)

    return x
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
      "strides": (1, 1), 
      "padding": "same",
      "use_bias": False # should we use bias ?
    }
    # 1x1 filter for conserving dimensions
    residual1x1 = k.layers.Conv2D(**def_1x1)(prev_layer)

    maps_kws = [
      dict(
        filters=int(W * i), 
        kernel_size=(3, 3), 
        activation="relu", 
        padding="same"
      ) for i in weights
    ]
    
    first  = k.layers.Conv2D(**maps_kws[0])(prev_layers)
    second = k.layers.Conv2D(**maps_kws[1])(first)
    third  = k.layers.Conv2D(**maps_kws[2])(second)

    # Concatenate successive 3x3 convolution maps :
    out = K.layers.Concatenate()([first, second, third])

    # And add the new 7x7 map with the 1x1 map, batch normalized
    out = K.layers.add([residual1x1, out])
    out = K.layers.Activation("relu")(out)

    return out
##


def ResPath(encoder_out, filter_size: int, n_filters: int):
    """
    """

    def_1x1  = {
      "filters": n_filters,
      "kernel_size": (1, 1), 
      "strides": (1, 1), 
      "padding": "same",
      "use_bias": False 
    }
    
    def_3x3 = {
      "filters": n_filters,
      "kernel_size": (3, 3), 
      "strides": (1, 1), 
      "activation": "relu", 
      "padding": "same",
      "use_bias": False 
    }

    x = k.layers.Conv2D(**def_1x1)(encoder_out)
    y = k.layers.Conv2D(**def_3x3)(encoder_out)
    y = keras.layers.add([x, y])

    for _ in range(filter_size - 1):
      pass
      #x = k.layers.add



