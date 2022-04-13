
import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU


class Pix2PixPreprocesamiento():
    def _init_(self):
        super().__init__()

    
    # definir un bloque codificador
    def definir_bloque_codificador(self,layer_in, n_filters, batchnorm=True):
	    # weight initialization
        init = RandomNormal(stddev=0.02)
	    # Inicialización del peso
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	    # añadir condicionalmente la normalización por lotes
        if batchnorm:
            g = BatchNormalization()(g, training=True)
	    # activación de la relu de la fuga
        g = LeakyReLU(alpha=0.2)(g)
        return g
 
 
    # definir un bloque decodificador
    def definir_bloque_decodificador(self,layer_in, skip_in, n_filters, dropout=True):
	    # Inicialización de pesos
        init = RandomNormal(stddev=0.02)
	    # se añade capa de upsampling
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # se añade normalización por lotes
        g = BatchNormalization()(g, training=True)
	    # añadir con condición el dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)
	    # fusionar con la conexión de salto
        g = Concatenate()([g, skip_in])
        # activación del relu
        g = Activation('relu')(g)
        return g
 
 
    # cargar y preparar las imágenes de entrenamiento
    def cargar_ejemplos_reales(self,filename):
	    # cargar matrices comprimidas
        data = load(filename)
	    # desempacar arreglos
        X, Y = data['arr_0'], data['arr_1']
	    # escala de [0,255] a [-1,1]
        #NORMALIZACION
        X = (X - 127.5) / 127.5
        Y = (Y - 127.5) / 127.5
        return [X, Y]
 
    # selecciona un lote de muestras aleatorias, devuelve las imágenes y el objetivo
    def generar_ejemplos_reales(self,dataset, n_samples, patch_shape):
	    # descomprimir dataset
        X, Y = dataset
	    # elegir una instancia al azar
        ix = randint(0, X.shape[0], n_samples)
	    # recuperar las imágenes seleccionadas
        delta_x, delta_y = X[ix], Y[ix]
	    # generar etiquetas de clase "reales" (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [delta_x, delta_y], y
 
 
    # generate a batch of images, returns images and targets
    def generar_ejemplos_falsos(self,g_model, samples, patch_shape):
    # generar una instancia falsa
        X = g_model.predict(samples)
	    # crear etiquetas de clase "falsas" (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y
 