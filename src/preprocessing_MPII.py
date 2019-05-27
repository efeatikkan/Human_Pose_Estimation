import cv2
from scipy.ndimage import rotate   
import numpy as np
from PIL import Image
import random
from scipy.ndimage.filters import gaussian_filter
import klepto.archives as klepto
from operator import itemgetter

random.seed(123)
archive_train=klepto.dir_archive('../dataset/MPII/Train',{},cached=False)
archive_test=klepto.dir_archive('../dataset/MPII/Test',{},cached=False)

# Size of heatmap and pictures
h_pic=256
w_pic=256
h_heat=64
w_heat=64

def split_train_test(data):
# =============================================================================
#     Split Train and Test
# =============================================================================
    train_id=random.sample(range(len(data)), int(len(data)*0.7))
    test_id=list(set(range(len(data)))-set(train_id))
    return train_id, test_id


def crop_pic(im, side,joint_positions ,center):
# =============================================================================
#     Crop and pad image
# =============================================================================
    upper_left_x=int(center[0] - side//2)
    bottom_right_x=int(center[0] + side//2)
    upper_left_y=int(center[1] +15 - side//2)
    bottom_left_y=int(center[1] +15 + side//2)
    
    if upper_left_x<0:upper_left_x=0
    if bottom_right_x<0:bottom_right_x=0
    if upper_left_y<0:upper_left_y=0
    if bottom_left_y<0:bottom_left_y=0

    # Calculate the new position of the joints
    new_coordinates=[]
    for j in joint_positions:
        if j[0]!=0 and j[1]!=0:     
            new_coordinates.append((int(j[0]-upper_left_x),int( j[1]-upper_left_y)))
        else:
            new_coordinates.append((0,0))
    im=im[upper_left_y:bottom_left_y, upper_left_x:bottom_right_x]
    return (im, new_coordinates)


def img_annonate_writer(dict_pic):
# ============================================================================
#     Take info of a picture
# =============================================================================
    imm= np.array(Image.open('../dataset/MPII/pictures/'+dict_pic['name']))

    joint_positions= dict_pic['joints']
    
    center= dict_pic['objpos']

    #coordinates and side length of bounding box
    side= dict_pic['scale'] *200 *1.25    

    
    return [imm,side,joint_positions, center]
        

def rot(image, joints):
# =============================================================================
#     Rotate a picture and calculate the new position of the joints
# =============================================================================

    angle=random.randint(-35,35)
    im_rot = rotate(image,angle) 
    
    # new joints coordinates
    org_center = (np.array(image.shape[:2][::-1])-1)//2
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)//2
    new=[]
    angle = np.deg2rad(angle)
    
    # Calculate the new position of the joints    
    for joint in joints:
        if joint[0]!=0 and joint[1]!=0:
            org=joint-org_center
            new.append([int(org[0]*np.cos(angle) + org[1]*np.sin(angle)+rot_center[0]),
              int(-org[0]*np.sin(angle) +org[1]*np.cos(angle)+rot_center[1])])
        else: 
            new.append([0,0])
            
    return im_rot, np.array(new)



def augmented_img(img,joints,dict_pic,name):
# =============================================================================
#     Augment a picture
# =============================================================================
    for num in range(random.randint(2,3)):
        name+=1
        img_ag,joints_ag=rot(img,joints)
        # Random gaussian filter
        if bool(random.getrandbits(1)):
            img_ag = gaussian_filter(img_ag,random.randint(1,4))
        
        img_ag,joints_ag=padder_resize(img_ag,joints_ag)
        heatmap=converting(joints_ag,img_ag.shape)
        
        # Resize picture (256,256)
        archive_train[str(name)+'a.jpg']={'img':img_ag,'joints':heatmap}

    return name
        
        
        
def converting(joints_coord,img_shape):
# =============================================================================
#     Calculate the new poistion of the joints after resizing
# =============================================================================
    my_dict_joints=[]

    ratio=h_heat/np.array([img_shape[1],img_shape[0]])
    new_joints=joints_coord*ratio
    for joint in new_joints:

        arr=np.zeros((h_heat,w_heat))
        
        # Set the information of unknown joints as equal to zero
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
    

def preprocess(dict_pic,name,flag=True):
# =============================================================================
#     Image preprocessing
# =============================================================================
    if flag:
        archive_=archive_train
    else:
        archive_=archive_test
    img,coordinates=crop_pic(*img_annonate_writer(dict_pic))
    
    pic,joints=padder_resize(img,coordinates)
    heatmap=converting(np.array(joints),pic.shape)

    
    archive_[str(name)+'.jpg']={'img':np.array(pic),'joints':heatmap}

    
    return (pic,joints) if flag else None

def padder_resize(img,coordinates):
    new_size=(h_pic,w_pic)
    pic=Image.fromarray(img)
    old_size=pic.size
    pic.thumbnail(new_size,Image.ANTIALIAS)

    ratio=np.array(pic.size)/np.array(old_size)
    coordinates=coordinates*ratio
    
    pic.thumbnail(new_size,Image.ANTIALIAS)
    img=cv2.copyMakeBorder(np.array(pic),top=int((new_size[1]-pic.size[1])/2),bottom=int((new_size[1]-pic.size[1])/2)+(new_size[1]-pic.size[1])%2,
                          left=int((new_size[0]-pic.size[0])/2),right=int((new_size[0]-pic.size[0])/2)+(new_size[0]-pic.size[0])%2,
                          borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
    
    new_coordinates=[]
    for j in coordinates:
        if j[0]!=0 and j[1]!=0:     
            new_coordinates.append((int(j[0]+(new_size[0]-pic.size[0])/2),int( j[1]+(new_size[1]-pic.size[1])/2)))
        else:
            new_coordinates.append((0,0))

    return img,new_coordinates

        
def create_data(id_pics,data,flag=True):
# =============================================================================
#     Creation of data
# =============================================================================
    
    name=0
    info_pics=itemgetter(*id_pics)(data)
    for dict_pic in info_pics:
        name+=1
        if flag:
            preprocess(dict_pic,name,False)
        else:
            img,coordinates=preprocess(dict_pic,name)
            name=augmented_img(img,np.array(coordinates), dict_pic,name)





