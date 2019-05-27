from PIL import Image
import numpy as np
import create_annotation as create
import cv2

pos_joints={0:'Right Ankle',1:'Right Knee',2:'Right Hip',3:'Left Hip',4:'Left Knee',5:'Left Ankle',
            6:'Pelvis',7:'Thorax',8:'Upper Neck',9:'Head Top',10:'Right Wrist',11:'Right Elbow',12:'Right Shoulder',13:'Left Shoulder',14:'Left Elbow',15:'Left Wrist'}

data=create.read_data('../labels/MPII/mpii_annotations.json')



for num in range(len(data)):
    pic=np.array(Image.open('../dataset/MPII/pictures/'+data[num]['name']))
    for i in data[num]['joints']:
        cv2.circle(pic,(int(i[0]),int(i[1])),3,(255,0,0),3)
    result = Image.fromarray(pic)
    result.save('../dataset/MPII/picturesjoints/'+str(num)+'.jpg')
    