import numpy as np
import sys
import matplotlib.pyplot as plt
from pix2pix_creacionmodelo import Pix2PixModeloGAN
from pix2pix_evaluarmodelo import Pix2PixEvaluarModelo
from pix2pix_generardataset import Pix2PixGenerarDataset
from pix2pix_preprocesamiento import Pix2PixPreprocesamiento


class Pix2PixEntrenamiento():
    def _init_(self):
        super().__init__()

 
   # entrenar modelos pix2pix
    def entrenarmodelo(self, d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	    # determinar la forma cuadrada de salida del discriminador
        d_acc_total_train = []
        d_loss_total_train = []
        d_acc_total_val = []
        d_loss_total_val = []
        d_acc_sum_train = []
        d_loss_sum_train = []
        d_acc_sum_val = []
        d_loss_sum_val = []
        n_patch = d_model.output_shape[1]
	    # descomprimir el dataset
        trainA, trainB = dataset
	    # calcula el número de lotes por época de entrenamiento 
        bat_per_epo = int(len(trainA) / n_batch)
	    # calcula el número de iteraciones de entrenamiento
        n_steps = bat_per_epo * n_epochs
        print(bat_per_epo)
        print(n_steps)
        # variables para la barra de carga por epoca
        bar_len = 60
        status=''
        # Enumerar manualmente las épocas
        for i in range(n_epochs):
            print('>Epoch '+str(i+1))
            for j in range(bat_per_epo):
                # seleccionar un lote de muestras reales
                [X_realA, X_realB], Y_real = Pix2PixPreprocesamiento().generar_ejemplos_reales(dataset, n_batch, n_patch)
	            # generar un lote de muestras falsas
                X_fakeB, y_fake = Pix2PixPreprocesamiento().generar_ejemplos_falsos(g_model, X_realA, n_patch)
                # actualizar los pesos en el discriminador para las muestras reales
                d_loss_train_real,d_acc_train_real = d_model.train_on_batch([X_realA, X_realB], Y_real)
                # actualizar los pesos en el discriminador para las muestras generadas
                d_acc_train_fake,d_loss_train_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
                # actualizar el generador
                g_loss, _, _ = gan_model.train_on_batch(X_realA, [Y_real, X_realB])
                
                if ((i+1) % 10 == 0):
                    d_loss_val_real, d_acc_val_real = Pix2PixEvaluarModelo().evaluar_acurracy(d_model, g_model, n_batch, n_patch)
                    d_acc_sum_train.append(d_acc_train_real)
                    d_loss_sum_train.append(d_loss_train_real)
                    d_acc_sum_val.append(d_acc_val_real)
                    d_loss_sum_val.append(d_loss_val_real)
                    
                filled_len = int(round(bar_len * (j+1) / float(bat_per_epo)))
                percents = round(100.0 * (j+1) / float(bat_per_epo), 1)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)
                sys.stdout.write('  >Entrenando [%s] %s%s ...%s\r' % (bar, percents, '%', status))
                sys.stdout.flush()
                
            if ((i+1) % 10 == 0):  
                d_acc_total_train.append(np.asarray(d_acc_sum_train).sum()/len(d_acc_sum_train))
                d_loss_total_train.append(np.asarray(d_loss_sum_train).sum()/len(d_loss_sum_train))
                d_acc_total_val.append(np.asarray(d_acc_sum_val).sum()/len(d_acc_sum_val))
                d_loss_total_val.append(np.asarray(d_loss_sum_val).sum()/len(d_loss_sum_val))
                Pix2PixEvaluarModelo().evaluar_performance((i), g_model, dataset)
                Pix2PixEvaluarModelo().guardar_modelo((i), g_model)
                self.graficar_accuracy_y_perdida( d_acc_total_train, d_loss_total_train, d_acc_total_val,d_loss_total_val, (i+1))
                file = open('acc.txt', 'w')
                for acc in d_acc_total_val:
                    file.writelines(str([acc]))
                file.close()
                d_acc_sum_train.clear()
                d_loss_sum_train.clear()
                d_acc_sum_val.clear()
                d_loss_sum_val.clear()
                filled_len = 0
                percents = 0
                status=''
                sys.stdout.write("\n")
                
            print('Termino epoca')
        print(d_acc_total_val)      
        
        
        '''         
        # Enumerar manualmente las épocas
        for i in range(n_steps):
		    # seleccionar un lote de muestras reales
            [X_realA, X_realB], Y_real = Pix2PixPreprocesamiento().generar_ejemplos_reales(dataset, n_batch, n_patch)
	        # generar un lote de muestras falsas
            X_fakeB, y_fake = Pix2PixPreprocesamiento().generar_ejemplos_falsos(g_model, X_realA, n_patch)
            # actualizar los pesos en el discriminador para las muestras reales
            d_loss_train_real,d_acc_train_real = d_model.train_on_batch([X_realA, X_realB], Y_real)
            # actualizar los pesos en el discriminador para las muestras generadas
            d_acc_train_fake,d_loss_train_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # actualizar el generador
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [Y_real, X_realB])
            # imprimir en consola el avance
                
            #print('>%d, Loss:[%.3f] Acc:[%.3f] Loss of gan[%.3f]' % (i+1,  d_loss_train_real, d_acc_train_real, g_loss))
		    # resumir el rendimiento del modelo
            if i == 0:
                print('Epoch 1')
            
            if (epoch % 2 == 0):
                d_loss_val_real, d_acc_val_real = Pix2PixEvaluarModelo().evaluar_acurracy(d_model, g_model, n_batch, n_patch)
                d_acc_sum_train.append(d_acc_train_real)
                d_loss_sum_train.append(d_loss_train_real)
                d_acc_sum_val.append(d_acc_val_real)
                d_loss_sum_val.append(d_loss_val_real)
            
            if( (i+1) % bat_per_epo  == 0):
                print(i)
                print('Termino iteracion')
                epoch = epoch + 1
                print('>Epoch '+str(epoch))
                count = 0
                filled_len = 0
                percents = 0
                status=''
                sys.stdout.write("\n")
                
            #EVALUACION
            if (n_steps) % 2 == 0:
                #promedio
                print(i)
                d_acc_total_train.append(np.asarray(d_acc_sum_train).sum()/len(d_acc_sum_train))
                d_loss_total_train.append(np.asarray(d_loss_sum_train).sum()/len(d_loss_sum_train))
                d_acc_total_val.append(np.asarray(d_acc_sum_val).sum()/len(d_acc_sum_val))
                d_loss_total_val.append(np.asarray(d_loss_sum_val).sum()/len(d_loss_sum_val))
                Pix2PixEvaluarModelo().evaluar_performance(epoch, g_model, dataset)
                Pix2PixEvaluarModelo().guardar_modelo(epoch, g_model)
                self.graficar_accuracy_y_perdida( d_acc_total_train, d_loss_total_train, d_acc_total_val,d_loss_total_val)
                file = open('acc.txt', 'w')
                for acc in d_acc_total_val:
                    file.writelines([acc])
                file.close()
                d_acc_sum_train.clear()
                d_loss_sum_train.clear()
                d_acc_sum_val.clear()
                d_loss_sum_val.clear()
            
            filled_len = int(round(bar_len * count / float(bat_per_epo)))
            percents = round(100.0 * count / float(bat_per_epo), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write('  >Entrenando [%s] %s%s ...%s\r' % (bar, percents, '%', status))
            sys.stdout.flush()       
            count = count + 1

     '''            
        
    def graficar_accuracy_y_perdida(self, d_acc_total_train, d_loss_total_train, d_acc_total_val,d_loss_total_val, epoch):
    
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(d_acc_total_train, label='Train Acurracy')
        plt.plot(d_acc_total_val, label='Validation Acurracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([0,1])
        plt.title('Training and validation Acurracy')

        plt.subplot(2, 1, 2)
        plt.plot(d_loss_total_train, label='Training Loss')
        plt.plot(d_loss_total_val, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.ylim([0,1.0])
        plt.title('Training and validation Loss')
        plt.xlabel('epoch')
        #plt.show()  
        
        plt.savefig(str(epoch) + 'acc_and_loss.png')  


if __name__ == "__main__":
    print('Inicio Entrenamiento Proyecto Pix2Pix')
    # cargar los datos de la imagen y reescalar (preprocesamiento)
    dataset = Pix2PixPreprocesamiento().cargar_ejemplos_reales('maps_256.npz')
    print('Dataset cargado', dataset[0].shape, dataset[1].shape)
    # definir la forma de entrada basada en el conjunto de datos cargados
    #(256,256,3)
    image_shape = dataset[0].shape[1:]
    print(image_shape)
    # definir los modelos
    modeloGANs = Pix2PixModeloGAN()
    d_model = modeloGANs.definir_discriminador(image_shape)
    print("Modelo discriminador generado correctamente")
    g_model = modeloGANs.definir_generador(image_shape)
    print("Modelo generador generado correctamente")
    gan_model = modeloGANs.definir_gan(g_model, d_model, image_shape)
    print("Modelo GANs generado correctamente")
    
    Pix2PixEntrenamiento().entrenarmodelo(d_model, g_model, gan_model, dataset)
    
