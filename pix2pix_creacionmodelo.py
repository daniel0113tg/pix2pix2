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
from pix2pix_preprocesamiento import Pix2PixPreprocesamiento


class Pix2PixModeloGAN():
    def _init_(self):
        super().__init__()



     # definir el modelo discriminante
    def definir_discriminador(self,image_shape):
	    # Inicialización del peso
        init = RandomNormal(stddev=0.02)
        # entrada de la imagen de origen
        in_src_image = Input(shape=image_shape)
	    # entrada de la imagen de destino
        in_target_image = Input(shape=image_shape)
	    # concatenar las imágenes por canales
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
	    # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
	    # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
      # penúltima capa de salida
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # salida del parche
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
	    # se define modelo
        model = Model([in_src_image, in_target_image], patch_out)
	    # se compila modelo
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5], metrics = ['accuracy'])
        return model


 
 
    # definir el modelo de generador autónomo
    def definir_generador(self,image_shape=(256,256,3)):
	    # Inicialización del peso
        init = RandomNormal(stddev=0.02)
        #entrada de imagen
        in_image = Input(shape=image_shape)
	    # modelo de codificador
        e1 = Pix2PixPreprocesamiento().definir_bloque_codificador(in_image, 64, batchnorm=False)
        e2 = Pix2PixPreprocesamiento().definir_bloque_codificador(e1, 128)
        e3 = Pix2PixPreprocesamiento().definir_bloque_codificador(e2, 256)
        e4 = Pix2PixPreprocesamiento().definir_bloque_codificador(e3, 512)
        e5 = Pix2PixPreprocesamiento().definir_bloque_codificador(e4, 512)
        e6 = Pix2PixPreprocesamiento().definir_bloque_codificador(e5, 512)
        e7 = Pix2PixPreprocesamiento().definir_bloque_codificador(e6, 512)
        # cuello de botella, sin norma de lote y relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
	    # modelo de decodificador
        d1 = Pix2PixPreprocesamiento().definir_bloque_decodificador(b, e7, 512)
        d2 = Pix2PixPreprocesamiento().definir_bloque_decodificador(d1, e6, 512)
        d3 = Pix2PixPreprocesamiento().definir_bloque_decodificador(d2, e5, 512)
        d4 = Pix2PixPreprocesamiento().definir_bloque_decodificador(d3, e4, 512, dropout=False)
        d5 = Pix2PixPreprocesamiento().definir_bloque_decodificador(d4, e3, 256, dropout=False)
        d6 = Pix2PixPreprocesamiento().definir_bloque_decodificador(d5, e2, 128, dropout=False)
        d7 = Pix2PixPreprocesamiento().definir_bloque_decodificador(d6, e1, 64, dropout=False)
	    # salida
        g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        # definir modelo
        model = Model(in_image, out_image)
        return model

    # definir el modelo combinado de generador y discriminador, para actualizar el generador
    def definir_gan(self,g_model, d_model, image_shape):
	    # hacer que los pesos del discriminador no sean entrenables
        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        in_src = Input(shape=image_shape)
	    # Conectar la imagen de origen a la entrada del generador
        gen_out = g_model(in_src)
	    # Conectar la entrada de la fuente y la salida del generador a la entrada del discriminador
        dis_out = d_model([in_src, gen_out])
	    # imagen src como entrada, imagen generada y salida de clasificación
        model = Model(in_src, [dis_out, gen_out])
	    # compilar el modelo
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
        return model
 
 

