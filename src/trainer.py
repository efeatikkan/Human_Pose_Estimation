import model
from keras.optimizers import RMSprop
import numpy as np
import klepto.archives as klepto
import random
from keras import backend as K 
import tensorflow as tf


random.seed(123)

# Size of heatmap and pictures
h_pic=256
w_pic=256
h_heat=64
w_heat=64

        

def train_data_generator(path,batch_size, inres=(h_pic,w_pic) , outres= (h_heat,w_heat)):
# =============================================================================
#     Create data generator 
# =============================================================================
    archive_train = klepto.dir_archive(path,cached=False)
    archive_train.load()
    all_images = np.array(list(archive_train.keys()))
    size = len(all_images)

    while True:
        
        # take random images
        names = np.random.permutation(list(archive_train.keys()))
        num_of_batches= size// batch_size 
        
        for im in range(num_of_batches): 
            gt_stack = np.zeros(shape=(batch_size, outres[0], outres[1],nOutput))
            img_stack= np.zeros(shape= (batch_size, inres[0], inres[1], 3))

            selected_photo_names = names[im * batch_size : (im+1)* batch_size]

            for j in range(len(selected_photo_names)):
                
                gt_stack[j,:,:,:] = np.transpose(np.array(archive_train[selected_photo_names[j]]['joints']),(1,2,0))
                img_stack[j,:,:,:] = archive_train[selected_photo_names[j]]['img']/255.
               
            yield(img_stack,{'stack_0':gt_stack,'stack_1':gt_stack})

def custom_loss_mse(y_true, y_pred):
        alpha=0.2
        l0= K.mean(K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,1] - y_pred[:,:,:,1]),axis=-1)+K.mean(K.square(y_true[:,:,:,2] - y_pred[:,:,:,2]),axis=-1))
        l1= K.mean(K.square(y_true[:,:,:,1] - y_pred[:,:,:,1]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]),axis=-1)+K.mean(K.square(y_true[:,:,:,2] - y_pred[:,:,:,2]),axis=-1))
        l2= K.mean(K.square(y_true[:,:,:,2] - y_pred[:,:,:,2]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]),axis=-1)+K.mean(K.square(y_true[:,:,:,1] - y_pred[:,:,:,1]),axis=-1))
        l3= K.mean(K.square(y_true[:,:,:,3] - y_pred[:,:,:,3]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,4] - y_pred[:,:,:,4]),axis=-1)+K.mean(K.square(y_true[:,:,:,5] - y_pred[:,:,:,5]),axis=-1))
        l4= K.mean(K.square(y_true[:,:,:,4] - y_pred[:,:,:,4]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,3] - y_pred[:,:,:,3]),axis=-1)+K.mean(K.square(y_true[:,:,:,5] - y_pred[:,:,:,5]),axis=-1))
        l5= K.mean(K.square(y_true[:,:,:,5] - y_pred[:,:,:,5]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,3] - y_pred[:,:,:,3]),axis=-1)+K.mean(K.square(y_true[:,:,:,4] - y_pred[:,:,:,4]),axis=-1))
        l6= K.mean(K.square(y_true[:,:,:,6] - y_pred[:,:,:,6]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,7] - y_pred[:,:,:,7]),axis=-1))
        l7= K.mean(K.square(y_true[:,:,:,7] - y_pred[:,:,:,7]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,6] - y_pred[:,:,:,6]),axis=-1))
        
        l8= K.mean(K.square(y_true[:,:,:,8] - y_pred[:,:,:,8]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,9] - y_pred[:,:,:,9]),axis=-1))
        l9= K.mean(K.square(y_true[:,:,:,9] - y_pred[:,:,:,9]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,8] - y_pred[:,:,:,8]),axis=-1))
        
        l10= K.mean(K.square(y_true[:,:,:,10] - y_pred[:,:,:,10]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,11] - y_pred[:,:,:,11]),axis=-1)+K.mean(K.square(y_true[:,:,:,12] - y_pred[:,:,:,12]),axis=-1))
        l11= K.mean(K.square(y_true[:,:,:,11] - y_pred[:,:,:,11]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,10] - y_pred[:,:,:,10]),axis=-1)+K.mean(K.square(y_true[:,:,:,12] - y_pred[:,:,:,12]),axis=-1))
        l12= K.mean(K.square(y_true[:,:,:,12] - y_pred[:,:,:,12]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,10] - y_pred[:,:,:,10]),axis=-1)+K.mean(K.square(y_true[:,:,:,11] - y_pred[:,:,:,11]),axis=-1))
        
        l13= K.mean(K.square(y_true[:,:,:,13] - y_pred[:,:,:,13]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,14] - y_pred[:,:,:,14]),axis=-1)+K.mean(K.square(y_true[:,:,:,15] - y_pred[:,:,:,15]),axis=-1))
        l14= K.mean(K.square(y_true[:,:,:,14] - y_pred[:,:,:,14]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,13] - y_pred[:,:,:,13]),axis=-1)+K.mean(K.square(y_true[:,:,:,15] - y_pred[:,:,:,15]),axis=-1))
        l15= K.mean(K.square(y_true[:,:,:,15] - y_pred[:,:,:,15]),axis=-1) + alpha* (K.mean(K.square(y_true[:,:,:,13] - y_pred[:,:,:,13]),axis=-1)+K.mean(K.square(y_true[:,:,:,14] - y_pred[:,:,:,14]),axis=-1))
        
        total_loss = l0 + l1 +l2 +l3 +l4 + l5 +l6 +l7 +l8 + l9 +l10 +l11 +l12 + l13 +l14 +l15
        return(total_loss)


def train_model(path_data,path_model,njoints,batch_size=6,optimizer=RMSprop(lr=5e-4),loss=custom_loss_mse,metrics=['accuracy'],epochs=100,step_epochs=800):
# =============================================================================
#     Train the model
# =============================================================================
    global nOutput
    nOutput=njoints
    
    cbk = tf.keras.callbacks.TensorBoard("logging/keras_model")

    mymodel= model.hg_train(nOutput)
    data_gen= train_data_generator(path_data,batch_size)
    mymodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    with open(path_model+'modelsummary.txt','w') as summary_:
        mymodel.summary(print_fn=lambda x: summary_.write(x + '\n'))

    history=mymodel.fit_generator(data_gen,step_epochs, epochs, callbacks=[cbk])
    model_json = mymodel.to_json()

    with open(path_model+"model.json","w") as json_file:
        json_file.write(model_json)

    mymodel.save_weights(path_model+"weight.h5")
    return history

