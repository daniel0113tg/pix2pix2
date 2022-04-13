# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
# load the prepared dataset
from numpy import load
from matplotlib import pyplot
# example of pix2pix gan for satellite to map image-to-image translation
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
from matplotlib import pyplot


class Pix2Pix():
    def _init_(self):
        super().__init__()

# load all images in a directory into memory
    def cargarimagenes(self,path, size=(256,512)):
        src_list, tar_list = list(), list()
	    # enumerate filenames in directory, assume all are images
        for filename in listdir(path):
            print(filename)
            # load and resize the image
            pixels = load_img(path + filename, target_size=size)
		    # convert to numpy array
            pixels = img_to_array(pixels)
		    # split into satellite and map
            sat_img, map_img = pixels[:, :256], pixels[:, 256:]
            src_list.append(sat_img)
            tar_list.append(map_img)
        print("Proceso completado")
        return [asarray(src_list), asarray(tar_list)]
    
    def cargardataset(self):
        data = load('maps_256.npz')
        src_images, tar_images = data['arr_0'], data['arr_1']
        print('Loaded: ', src_images.shape, tar_images.shape)
        # plot source images
        n_samples = 3
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(src_images[i].astype('uint8'))
        # plot target image
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(tar_images[i].astype('uint8'))
        pyplot.show()
        return src_images, tar_images

     # definir el modelo discriminante
    def define_discriminator(self,image_shape):
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
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model


    # definir un bloque codificador
    def define_encoder_block(self,layer_in, n_filters, batchnorm=True):
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
    def decoder_block(self,layer_in, skip_in, n_filters, dropout=True):
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
 
 
    # definir el modelo de generador autónomo
    def define_generator(self,image_shape=(256,256,3)):
	    # Inicialización del peso
        init = RandomNormal(stddev=0.02)
        #entrada de imagen
        in_image = Input(shape=image_shape)
	    # modelo de codificador
        e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)
        # cuello de botella, sin norma de lote y relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
	    # modelo de decodificador
        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
	    # salida
        g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        # definir modelo
        model = Model(in_image, out_image)
        return model

    # definir el modelo combinado de generador y discriminador, para actualizar el generador
    def define_gan(self,g_model, d_model, image_shape):
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
 
    # cargar y preparar las imágenes de entrenamiento
    def load_real_samples(self,filename):
	    # cargar matrices comprimidas
        data = load(filename)
	    # desempacar arreglos
        X1, X2 = data['arr_0'], data['arr_1']
	    # escala de [0,255] a [-1,1]
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5
        return [X1, X2]
 
    # selecciona un lote de muestras aleatorias, devuelve las imágenes y el objetivo
    def generate_real_samples(self,dataset, n_samples, patch_shape):
	    # descomprimir dataset
        trainA, trainB = dataset
	    # elegir una instancia al azar
        ix = randint(0, trainA.shape[0], n_samples)
	    # recuperar las imágenes seleccionadas
        X1, X2 = trainA[ix], trainB[ix]
	    # generar etiquetas de clase "reales" (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y
 
 
    # generate a batch of images, returns images and targets
    def generate_fake_samples(self,g_model, samples, patch_shape):
    # generar una instancia falsa
        X = g_model.predict(samples)
	    # crear etiquetas de clase "falsas" (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y
 
    # generate samples and save as a plot and save the model
    def summarize_performance(self,step, g_model, dataset, n_samples=3):
    # generar muestras y guardar como un gráfico y guardar el modelo
        [X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
	    # generar un lote de muestras falsas
        X_fakeB, _ = self.generate_fake_samples(g_model, X_realA, 1)
        # escalar todos los píxeles de [-1,1] a [0,1]
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0
	    # trazar imágenes de origen real
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realA[i])
	    # trazar la imagen objetivo generada
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_fakeB[i])
	    # trazar la imagen real del objetivo
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)    
            pyplot.axis('off')
            pyplot.imshow(X_realB[i])
	    # guardar la trama en un archivo
        filename1 = 'plot_%06d.png' % (step+1)
        pyplot.savefig(filename1)
        pyplot.close()
	    # guardar el modelo del generador
        filename2 = 'model_%06d.h5' % (step+1)
        g_model.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))
 
   # entrenar modelos pix2pix
    def train(self, d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	    # determinar la forma cuadrada de salida del discriminador
        n_patch = d_model.output_shape[1]
	    # descomprimir el dataset
        trainA, trainB = dataset
	    # calcula el número de lotes por época de entrenamiento 
        bat_per_epo = int(len(trainA) / n_batch)
	    # calcula el número de iteraciones de entrenamiento
        n_steps = bat_per_epo * n_epochs
        print(n_steps)
	# Enumerar manualmente las épocas
        for i in range(n_steps):
		    # seleccionar un lote de muestras reales
            [X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
	        # generar un lote de muestras falsas
            X_fakeB, y_fake = self.generate_fake_samples(g_model, X_realA, n_patch)
            # actualizar el discriminador para las muestras reales
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # actualizar el discriminador para las muestras generadas
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # actualizar el generador
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # resumir el rendimiento
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		    # resumir el rendimiento del modelo
            if (i+1) % (bat_per_epo * 10) == 0:
                self.summarize_performance(i, g_model, dataset)
 


if __name__ == "__main__":
    # dataset path
    #path = 'maps/train/'
    #print(path)
    # load dataset
    pix2pix = Pix2Pix()
    #[src_images, tar_images] = pix2pix.cargarimagenes(path)
    #print('Loaded: ', src_images.shape, tar_images.shape)
    # save as compressed numpy array
    ### savez_compressed(filename, src_images, tar_images)
    #filename = 'maps_256'
    #np.savez( filename, src_images, tar_images)
    #print('Dataset guardado: ', filename)
    print('Inicio')
    # cargar los datos de la imagen
    dataset = pix2pix.load_real_samples('maps_256.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)
    # definir la forma de entrada basada en el conjunto de datos cargados
    image_shape = dataset[0].shape[1:]
    # definir los modelos
    d_model = pix2pix.define_discriminator(image_shape)
    g_model = pix2pix.define_generator(image_shape)
    # definir el modelo compuesto
    gan_model = pix2pix.define_gan(g_model, d_model, image_shape)
    # entrenar modelo
    pix2pix.train(d_model, g_model, gan_model, dataset)