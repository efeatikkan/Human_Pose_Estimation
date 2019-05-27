import numpy as np
import math
from scipy.ndimage import maximum_filter
import cv2
from random import randint as rand
import matplotlib.pyplot as plt
import klepto.archives as klepto
import model

h_pic=256
w_pic=256
h_heat=64
w_heat=64



def load_model():
# =============================================================================
#     Import Model and weights
# =============================================================================
    mymodel=model.hg_train(16)
    mymodel.load_weights('../model/MPII/weight.h5')
    
    return mymodel


def argmax_(img):
# =============================================================================
#     Obtain the coordinates of the largest element in a matrix
# =============================================================================
    coordinates = np.unravel_index(np.argmax(img, axis=None), img.shape)
    return coordinates


def euclidean_dist(pred, gt_img, flag=True):
# =============================================================================
#     Euclidean distance 
# =============================================================================
    if flag:
        pred=non_max_suppression(pred)
    img_x, img_y = argmax_(pred)
    gt_img_x, gt_img_y = argmax_(gt_img)
    return math.sqrt((float(img_x - gt_img_x))**2 + (float(img_y - gt_img_y))**2)


def accuracy_pred(predictions, gt_maps,accurate_preds, num_pics):
# =============================================================================
#     Define the accuracy of a prediction based on defined threshold
# =============================================================================                               
    for preds in range(predictions[1].shape[3]):
        
        # exclude pics without upper neck or top head
        if all(np.unique(gt_maps[8]) == 0) or all(np.unique(gt_maps[9]) == 0):
            return accurate_preds, num_pics
        
        threshold = euclidean_dist(gt_maps[8],gt_maps[9],False)*0.5
                                   
        if all(np.unique(gt_maps[preds])==0):
            continue
            
        elif euclidean_dist(predictions[1][0,:,:,preds], gt_maps[preds]) <= threshold:
            accurate_preds[preds]+=1
        
        # count number of pics that have information on that particular joint
        num_pics[preds]+=1
        
    return accurate_preds, num_pics
    

    
def PCK(path_GT,path_pred,njoints):
# =============================================================================
#     Compute the PCK metric
# =============================================================================
    prediction_set = klepto.dir_archive(path_pred,cached=False)
    prediction_set.load()

    gt_maps = klepto.dir_archive(path_GT,cached=False)
    gt_maps.load()

    accuracy=[0]*njoints
    num_pics = [0]*njoints # number of pictures per joint
    
    for name in prediction_set.keys():
        accuracy,num_pics=accuracy_pred(prediction_set[name], gt_maps[name]['joints'],accuracy, num_pics)

    return np.array(accuracy)/np.array(num_pics)


def non_max_suppression(heatmap, windowSize=3, threshold=1e-3):
# =============================================================================
#     Compute Non-maximum Suppression
# =============================================================================

    # clear values less than threshold
    indices_under_thres = heatmap < threshold
    heatmap[indices_under_thres] = 0

    return heatmap * (heatmap == maximum_filter(heatmap, footprint=np.ones((windowSize, windowSize))))



def prediction(path_GT,path_pred,mymodel):
# =============================================================================
#     Compute Prediction of image
# =============================================================================
    prediction=klepto.dir_archive(path_pred,{},cached=False)

    archive= klepto.dir_archive(path_GT,cached=False)
    archive.load()

    for name in archive.keys():
        img=archive[name]['img'].reshape(1,w_pic,h_pic,3)
        predict_heat=mymodel.predict(img/255)
        prediction[name]=predict_heat


def rescale(coords):
# =============================================================================
#     Transform coordinates into 256x256 space 
# =============================================================================
    x=(w_pic*coords[1])//w_heat
    y=(h_pic*coords[0])//h_heat
    return (x,y)


def rescale_joint_coords(heatmap):
# =============================================================================
#     Rescale joint coordinates
# =============================================================================
    for pos in range(heatmap.shape[3]):
        x,y = rescale(argmax_(heatmap[0,:,:,pos]))
        yield x,y


def draw_skeleton(path_GT,path_pred,path_visual,name,MPII):
# =============================================================================
#     Draw skeleton based on model predictions
# =============================================================================
    
    prediction=klepto.dir_archive(path_pred,cached=False)
    prediction.load()
    
    archive=klepto.dir_archive(path_GT,cached=False)
    archive.load()
    
    img=archive[name]['img'].astype('uint8')
    heatmap=prediction[name][1]
    
    # define connections between joints for each dataset
    if MPII:
        lines = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(12,7),(7,13),(13,14),(14,15)]
    else:
        lines = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11),(2,8),(3,9),(12,13)]
    coords = dict(enumerate(list(rescale_joint_coords(heatmap))))
    for points in lines:
        if coords[points[0]]==(0,0) or coords[points[1]]==(0,0): continue
        else:
            cv2.line(img, coords[points[0]], coords[points[1]], (rand(0,255),rand(0,255),rand(0,255)), thickness=2, lineType=8)
    plt.imshow(img)
    plt.imsave(path_visual+'Skeleton.png',img)

