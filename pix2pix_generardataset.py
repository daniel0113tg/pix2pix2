# Cargar, dividir y escalar el conjunto de datos de los mapas listo para el entrenamiento
from os import listdir
from numpy import asarray
from numpy import vstack
import numpy as np
from numpy import savez_compressed
# cargar el conjunto de datos preparado
from numpy import load
from matplotlib import pyplot


class Pix2PixGenerarDataset():
    def _init_(self):
        super().__init__()

# cargar todas las imágenes de un directorio en la memoria
    def cargarimagenes(self,path, size=(256,512)):
        src_list, tar_list = list(), list()
	    # Enumerar los nombres de archivos en el directorio, asumir que todos son imágenes
        for filename in listdir(path):
            print(filename)
            # cargar y redimensionar la imagen
            pixels = load_img(path + filename, target_size=size)
		    # convertir en matriz numpy
            pixels = img_to_array(pixels)
		    # dividir en imagen de satélite y mapa
            sat_img, map_img = pixels[:, :256], pixels[:, 256:]
            src_list.append(sat_img)
            tar_list.append(map_img)
        print("Proceso completado")
        return [asarray(src_list), asarray(tar_list)]
    
    def cargardataset(self):
        data = load('maps_256.npz')
        src_images, tar_images = data['arr_0'], data['arr_1']
        print('Loaded: ', src_images.shape, tar_images.shape)
        # dibujar las imágenes de origen
        n_samples = 3
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(src_images[i].astype('uint8'))
        # dibujar la imagen de destino
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(tar_images[i].astype('uint8'))
        pyplot.show()
        return src_images, tar_images
        # (226,256,3)
        
    def cargardataset_val(self):
        data = load('maps_256_val.npz')
        src_images, tar_images = data['arr_0'], data['arr_1']
        print('Loaded: ', src_images.shape, tar_images.shape)
        # dibujar las imágenes de origen
        n_samples = 3
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(src_images[i].astype('uint8'))
        # dibujar la imagen de destino
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(tar_images[i].astype('uint8'))
        pyplot.show()
        return src_images, tar_images
        # (226,256,3)        
        

 
if __name__ == "__main__":
    # cargar imagenes 
    path = 'maps/train/'
    pix2pix = Pix2PixGenerarDataset()
    # ruta del dataset 
    print(path)
    [src_images, tar_images] = pix2pix.cargarimagenes(path)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # guardar como matriz numpy comprimida
    filename = 'maps_256'
    savez_compressed(filename, src_images, tar_images)
    np.savez( filename, src_images, tar_images)
    print('Dataset guardado: ', filename)
    
    path = 'maps/val/'
    # ruta del dataset 
    print(path)
    [src_images, tar_images] = pix2pix.cargarimagenes(path)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # guardar como matriz numpy comprimida
    filename = 'maps_256_val'
    savez_compressed(filename, src_images, tar_images)
    np.savez( filename, src_images, tar_images)
    print('Dataset guardado: ', filename)
