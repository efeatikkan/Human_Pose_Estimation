import json
from PIL import Image


def read_data(json_file_source):
# =============================================================================
#     Read Json file
# =============================================================================
    with open(json_file_source) as f:
        all_data= json.load(f)
    return(all_data)
    
def find_single_person(data):
# =============================================================================
#     Find pictures where people are separable
# =============================================================================
    src_dir = "../dataset/MPII/all_pictures/"
    dst_dir = "../dataset/MPII/pictures/"

    for pos in range(len(data['RELEASE']['single_person'])):
        if data['RELEASE']['single_person'][pos]==[] or data['RELEASE']['img_train'][pos]==0:
            continue
        else:
            name=data['RELEASE']['annolist'][pos]['image']['name']
            pic=Image.open(src_dir+name)
            pic.save(dst_dir+name)


def list_info_pictures(data):
# =============================================================================
#     List of dictionaries, which contain info about each picture
# =============================================================================
    list_info=[]
    id_person=[]
    for pos in range(len(data['RELEASE']['single_person'])):
        if data['RELEASE']['single_person'][pos]==[] or data['RELEASE']['img_train'][pos]==0:
            continue
        else:
            id_person.append(data['RELEASE']['single_person'][pos])
            list_info.append(data['RELEASE']['annolist'][pos])
    return list_info,id_person
            

def define_joints(data):
# =============================================================================
#     Add occluded joints
# =============================================================================
    joints=[[0,0,[]]]*16
    list_joints=sorted(data,key=lambda dicty:dicty['id'])

    for id_joint in range(len(list_joints)):
        if list_joints[id_joint]['is_visible']==[]:
            joints[list_joints[id_joint]['id']]=[list_joints[id_joint]['x'],list_joints[id_joint]['y'],list_joints[id_joint]['is_visible']]
        else:
            joints[list_joints[id_joint]['id']]=[list_joints[id_joint]['x'],list_joints[id_joint]['y'],bool(int(list_joints[id_joint]['is_visible']))]

    return joints


def write_json(data_info):
# =============================================================================
#     Create json file
# =============================================================================
    data,id_people=list_info_pictures(data_info)

    json_list=[]
    for pos in range(len(data)):
        dicty_info=data[pos]
        if type(dicty_info['annorect']) is dict:

            info={}
            info['name']=dicty_info['image']['name']
            info['objpos']=[dicty_info['annorect']['objpos']['x'],dicty_info['annorect']['objpos']['y']]
            info['scale']=dicty_info['annorect']['scale']
            info['id_person']=0
            info['joints']=define_joints(dicty_info['annorect']['annopoints']['point'])
            json_list.append(info)
        elif type(dicty_info['annorect']) is list:

            if type(id_people[pos]) is list:
                id_pep=id_people[pos]
                
            else:
                id_pep=[id_people[pos]]

            for person_id in id_pep:
                if dicty_info['annorect'][person_id-1]['annopoints']==[] or type(dicty_info['annorect'][person_id-1]['annopoints']['point']) is dict:
                    continue
                info={}
                info['name']=dicty_info['image']['name']
                info['objpos']=[dicty_info['annorect'][person_id-1]['objpos']['x'],dicty_info['annorect'][person_id-1]['objpos']['y']]
                info['scale']=dicty_info['annorect'][person_id-1]['scale']
                info['id_person']=person_id
                info['joints']=define_joints(dicty_info['annorect'][person_id-1]['annopoints']['point'])
                json_list.append(info)

    with open('../labels/MPII/mpii_annotations.json', 'w') as file:
        json.dump(json_list, file)
                

            

