import os
import cv2
from scipy.ndimage import rotate   
import numpy as np
from PIL import Image
import random
from scipy.ndimage.filters import gaussian_filter
import klepto.archives as klepto

random.seed(123)
archive_train=klepto.dir_archive('../dataset/UP14/Train',{},cached=False)
archive_test=klepto.dir_archive('../dataset/UP14/Test',{},cached=False)

# Size of heatmap and pictures
h_pic=256
w_pic=256
h_heat=64
w_heat=64

def read_name(path_file_source):
# =============================================================================
#     Read Json file
# =============================================================================
    basepath = path_file_source
    name_pic=[name[:5] for name in os.listdir(basepath)]
    train_name=random.sample(name_pic, int(len(name_pic)*0.8))
    test_name=list(set(name_pic)-set(train_name))
    return train_name, test_name


def padder(im, joint_positions):
# =============================================================================
#     Crop and pad an image; Calcultate the new position of the joints
# =============================================================================
    size_old=im.size
    im.thumbnail((w_pic,h_pic),Image.ANTIALIAS)
    size=im.size
    height=h_pic-size[1]
    width=w_pic-size[0]
    
    
    im=cv2.copyMakeBorder(np.array(im),top=height//2,bottom=height//2+size[1]%2,left=width//2+size[0]%2,
                          right=width//2,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
        
    new_coordinates=[]
    for j in joint_positions:
        if j[0]!=0 and j[1]!=0:     
            new_coordinates.append(((int(j[0])*size[0])/size_old[0]+width//2+size[0]%2,
                                    (int(j[1])*size[1])/size_old[1]+height//2+size[1]%2))
        else:
            new_coordinates.append((0,0))
    return im, new_coordinates

        

def rot(image, joints):
# =============================================================================
#     Rotate a picture and calculate the new position of the joints
# =============================================================================
    #random rotation angle
    angle=random.randint(-50,50)
    # rotate picture
    im_rot = rotate(image,angle) 
    # new joints coordinates
    org_center = (np.array(image.shape[:2][::-1])-1)//2
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)//2
    new=[]
    angle = np.deg2rad(angle)
    #find the new positions of the joints
    for joint in joints:
        if joint[0]!=0 and joint[1]!=0:
            org=joint-org_center
            new.append([int(org[0]*np.cos(angle) + org[1]*np.sin(angle)+rot_center[0]),
              int(-org[0]*np.sin(angle) +org[1]*np.cos(angle)+rot_center[1])])
        else: 
            new.append([0,0])

    return im_rot, np.array(new)


# augment the dataset 
def augmented_img(img,joints,name):
# =============================================================================
#     Augment a picture
# =============================================================================
    for num in range(random.randint(2,3)):
        name+=1
        img_ag,joints_ag=rot(img,joints)
        # Random gaussian filter
        if bool(random.getrandbits(1)):
            img_ag = gaussian_filter(img_ag,random.randint(1,4))
        heatmap=converting(joints_ag,img_ag.shape)
        pic=Image.fromarray(img_ag)
        pic=pic.resize((h_pic,w_pic), Image.ANTIALIAS)
        archive_train[str(name)+'a.jpg']={'img':np.array(pic),'joints':heatmap}
    return name
        
        
        
def converting(joints_coord,img_shape):
# =============================================================================
#     Calculate the new poistion of the joints after resizing
# =============================================================================
    my_dict_joints=[]
    # ratio
    ratio=h_heat/np.array([img_shape[1],img_shape[0]])
    new_joints=joints_coord*ratio
    for joint in new_joints:
        # add condition zero
        arr=np.zeros((h_heat,w_heat))
        # here we are going to consider the joint as it doesn't exist
        if joint[0]>63 or joint[1]>63 or joint[0]<0 or joint[1]<0:
            joint=[0,0]

        if joint[0]==0 and joint[1]==0:
            arr=arr
        else:
            arr[int(joint[1]),int(joint[0])]=1.
        # Gaussian peak on the joint
        arr=gaussian_filter(arr,1)
        my_dict_joints.append(arr)
    return my_dict_joints
    

def preprocess(image,joint,name,flag=True):
# =============================================================================
#     Image preprocessing
# =============================================================================
    if flag:
        archive_=archive_train
    else:
        archive_=archive_test
    img,coordinates=padder(image,joint)
    
    heatmap=converting(np.array(coordinates),img.shape)
    archive_[str(name)+'.jpg']={'img':img,'joints':heatmap}
    return img,coordinates if flag else None

        
def mirrored(image,joint,name):
    name+=1
    image=np.fliplr(np.array(image))
    image=Image.fromarray(image)
    img,coordinates=padder(image,joint)
    new=[]
    for joint in coordinates:
        if joint[0]!=0 and joint[1]!=0:
            new.append((img.shape[0]-joint[0]-1,joint[1]))
        else: 
            new.append([0,0])
    new_mirr=new.copy()
    new[0]=new_mirr[5]
    new[1]=new_mirr[4]
    new[2]=new_mirr[3]
    new[3]=new_mirr[2]
    new[4]=new_mirr[1]
    new[5]=new_mirr[0]
    new[6]=new_mirr[11]
    new[7]=new_mirr[10]
    new[8]=new_mirr[9]
    new[9]=new_mirr[8]
    new[10]=new_mirr[7]
    new[11]=new_mirr[6]
    
    heatmap=converting(np.array(new),img.shape)
    archive_train[str(name)+'m.jpg']={'img':img,'joints':heatmap}
    return img,new,name



def create_data(data,flag=True):
# =============================================================================
#     Split train and test
# =============================================================================
    name=0
    for pic_name in data:
        name+=1
        image=Image.open("../dataset/UP14/pictures/"+pic_name+"_image.png")
        joint=np.load("../labels/UP14/"+pic_name+"_joints.npy")
        joint=list(zip(joint[0],joint[1]))
        if flag==False:
            preprocess(image,joint,name,False)
        else:
            img,coordinates=preprocess(image.copy(),joint.copy(),name)
            name=augmented_img(img,np.array(coordinates),name)
            
            img_mirr,coordinates_mirr,name=mirrored(image.copy(),joint.copy(),name)
            name=augmented_img(img_mirr,np.array(coordinates_mirr),name)

