"""
    Universidad de Guanajuato
    Division de Ciencias e Ingenierias
    Grupo de investigacion : DCI-Net
    Investigador responsable : Dr. Carlos Padierna
    Alumno : Gustavo Magaña Lopez

        MultiResUNet 
        Implementacion basada en el siguiente articulo :
        https://arxiv.org/abs/1902.04049
"""

from typing import List

from tensorflow import keras as k
from sklearn.metrics import jaccard_score as jaccard_index

from timing import time_this

def MultiResBlock(prev_layer, U: int, alpha: float = 1.67, weights: List[float] = [0.167, 0.333, 0.5]):
    """
      As defined in the paper, with the possibility of 
      changing the weights of each one of the three successive
      convolutional layers.

      W is the number of filters of the convolutional layers 
      inside of the MultiResBlock, calculated from 'U' and 'alpha'
      as follows:

        W = alpha * U
      
      Arguments:
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
            
 
      Returns:
          out: A Keras layer.

    """
    
    W = alpha * U
    
    1x1_def  = {
      "filters": sum(map(lambda x: int(W * x), weights)), 
      "kernel_size": (1, 1), 
      "strides": (1, 1), 
      "padding": "same",
      "use_bias": False # should we use bias ?
    }
    # 1x1 filter for conserving dimensions
    residual1x1 = k.layers.Conv2D(**1x1_def)(prev_layer)

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

    1x1_def  = {
      "filters": n_filters,
      "kernel_size": (1, 1), 
      "strides": (1, 1), 
      "padding": "same",
      "use_bias": False # should we use bias ?
    }
    
    3x3_def = {
      "filters": n_filters,
      "kernel_size": (3, 3), 
      "strides": (1, 1), 
      "activation": "relu", 
      "padding": "same",
      "use_bias": False # should we use bias ?
    }

    x = k.layers.Conv2D(**1x1_def)(encoder_out)
    y = k.layers.Conv2D(**3x3_def)(encoder_out)
    y = keras.layers.add([x, y])

    for _ in range(filter_size - 1):
      x = k.layers.add



