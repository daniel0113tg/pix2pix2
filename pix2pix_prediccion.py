
# ejemplo de carga de un modelo pix2pix y su uso para la traducción de imágenes puntuales
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from keras.optimizers import Adam
from os import listdir
from numpy import vstack
import os
 
#cargar una imagen
def load_image(filename, size=(256,256)):
	# cargar la imagen con el tamaño preferido
	pixels = load_img(filename, target_size=size)
	# convertir en matriz numpy
	pixels = img_to_array(pixels)
	# escala de [0,255] a [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reajustar a 1 muestra
	pixels = expand_dims(pixels, 0)
	return pixels
 
# cargar la imagen de origen

path = 'maps/val'
titles = ['Source', 'Generated']
        
for filename in listdir(path):
        os.chdir('/Users/danieltacogallardo/Documentos/Projects/pix2pix')
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        model = load_model('models/model_024200.h5')
        gen_image = model.predict(src_image)
        images = vstack((src_image, gen_image))
        images = (images + 1) / 2.0
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])
            
        final = len(filename) -4
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('validation_model_024200')
        pyplot.savefig(filename1)

os.chdir('/Users/danieltacogallardo/Documentos/Projects/pix2pix')

for filename in listdir(path):    
        os.chdir('/Users/danieltacogallardo/Documentos/Projects/pix2pix')
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        model = load_model('models/model_043560.h5')
        gen_image = model.predict(src_image)
        images = vstack((src_image, gen_image))
        images = (images + 1) / 2.0
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])
            
        final = len(filename) -4
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('validation_model_043560')
        pyplot.savefig(filename1)
  

os.chdir('/Users/danieltacogallardo/Documentos/Projects/pix2pix')
for filename in listdir(path):    
        os.chdir('/Users/danieltacogallardo/Documentos/Projects/pix2pix')
        src_image = load_image(path + '/' + filename)
        print('Loaded', src_image.shape)
        model = load_model('models/model_009680.h5')
        gen_image = model.predict(src_image)
        #gen_image = (gen_image + 1) / 2.0
        images = vstack((src_image, gen_image))
        images = (images + 1) / 2.0
        for i in range(len(images)):
            pyplot.subplot(1, 2, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])
            
        final = len(filename) -4
        print(final)
        filename1 = 'plot_'+filename[0:len(filename)-4:1]+'.png'
        os.chdir('validation_model_009680')
        pyplot.savefig(filename1)
  

pyplot.close()


