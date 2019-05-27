import preprocessing_MPII as prep_MPII
import preprocessing_UP14 as prep_UP14
from trainer import train_model
import evaluation as eval_
import matplotlib.pyplot as plt
import numpy as np
import create_annotation as create

dataset ='MPII'

#Dataset Choice and Preprocessing
if dataset=='MPII':
    
    path_data_Train='../dataset/MPII/Train'
    path_data_Test='../dataset/MPII/Test'

    path_model='../model/MPII/'
    path_visual='../results/MPII/Visualization/'
    path_pred_Train='../results/MPII/Train'
    path_pred_Test='../results/MPII/Test'

    MPII=True
    num_joints=16

    data=create.read_data('../labels/MPII/mpii_annotations.json')
    train_id,test_id=prep_MPII.split_train_test(data)

    
    prep_MPII.create_data(train_id,data,False)
    prep_MPII.create_data(test_id,data,True)


elif dataset== 'UP14':
    path_data_Train='../dataset/UP14/Train'
    path_data_Test='../dataset/UP14/Test'

    path_model='../model/UP14/'
    path_visual='../results/UP14/Visualization/'
    path_pred_Train='../results/UP14/Train'
    path_pred_Test='../results/UP14/Test'
    
    MPII=False
    num_joints=14

    train_name,test_name=prep_UP14.read_name('../dataset/UP14/pictures')
    
    prep_UP14.create_data(train_name)
    prep_UP14.create_data(test_name,False)


#Train the model
history=train_model(path_data_Train,path_model,njoints=num_joints)

plt.plot(history.history['loss'])
plt.title('Loss at Training Time')
#plt.ylim(0.00001,0.000020)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(path_visual+'Loss.png')

#load model
mymodel=eval_.load_model()


# Prediction Train
eval_.prediction(path_data_Train,path_pred_Train,mymodel)   

#Evaluation Train
train_pck=eval_.PCK(path_data_Train,path_pred_Train,num_joints)
average_pck_train=np.mean(train_pck)

eval_.draw_skeleton(path_data_Train,path_pred_Train,path_visual,'43.jpg',MPII)





# Prediction Test
eval_.prediction(path_data_Test,path_pred_Test,mymodel)   

#Evaluation Test
test_pck=eval_.PCK(path_data_Test,path_pred_Test,num_joints)
average_pck_test=np.mean(test_pck)



#eval_.draw_skeleton(path_data_Test,path_pred_Test,path_visual,'917.jpg',MPII)





