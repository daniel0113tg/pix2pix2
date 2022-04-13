from numpy import load
from matplotlib import pyplot
from pix2pix_preprocesamiento import Pix2PixPreprocesamiento

class Pix2PixEvaluarModelo():
    def _init_(self):
        super().__init__()
 
    # generate samples and save as a plot and save the model
    def evaluar_performance(self,step, g_model, dataset, n_samples=3):
    # generar muestras y guardar como un gráfico y guardar el modelo
        [X_realA, X_realB], _ = Pix2PixPreprocesamiento().generar_ejemplos_reales(dataset, n_samples, 1)
	    # generar un lote de muestras falsas
        X_fakeB, _ = Pix2PixPreprocesamiento().generar_ejemplos_falsos(g_model, X_realA, 1)
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
 
    def evaluar_acurracy(self, d_model,  g_model, n_batch, n_patch):
        # Generamos nuevos datos
        dataset = Pix2PixPreprocesamiento().cargar_ejemplos_reales('maps_256_val.npz')
        [X_realA, X_realB], Y_real= Pix2PixPreprocesamiento().generar_ejemplos_reales(dataset, n_batch, n_patch)
        X_fakeB, y_fake =  Pix2PixPreprocesamiento().generar_ejemplos_falsos(g_model, X_realA, n_patch)

        # Evaluamos el modelo
        print('Real')
        loss_real, acc_real = d_model.evaluate([X_realA, X_realB], Y_real)
        print(d_model.metrics_names)
        print('Fake')
        loss_fake, acc_fake = d_model.evaluate([X_realA, X_fakeB], y_fake)
        return loss_real, acc_real

    def guardar_modelo(self, epoch, g_model):
        # guardar el modelo del generador
        filename2 = 'model_%d.h5' % (epoch+1)
        g_model.save(filename2)
        print('>Saved: %s' % (filename2))