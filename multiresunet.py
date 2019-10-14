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
                      dir(k.activations)[10:]
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



