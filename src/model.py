from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Input, Add, MaxPool2D, Activation, Concatenate, Lambda ,Softmax, ThresholdedReLU    
from tensorflow.keras.models import Model
import random
import tensorflow as tf

random.seed(123)
nFeatures = 256
nStack = 2 # number of hourglasses
nModules = 3 # number of residual modules


# =============================================================================
#     Architecture of single residual block:
#           - branch 1
#           - branch 2
#           - branch 3
#           - skip layer 
# =============================================================================
# =============================================================================
def  branch_1(inp):

    node = BatchNormalization()(inp)
    node = Activation(activation='relu')(node)
    node = Conv2D(32, (1, 1), padding='same')(node)   
    
    return node


def branch_2(inp):
    
    node = BatchNormalization()(inp)
    node = Activation(activation='relu')(node)
    node = Conv2D(32, (1, 1), padding='same')(node)   

    node = BatchNormalization()(node)
    node = Activation(activation='relu')(node)
    node = Conv2D(32, (3, 3), padding='same')(node)

    return node


def branch_3(inp):
    
    node = BatchNormalization()(inp)
    node = Activation(activation='relu')(node)
    node = Conv2D(32, (1, 1), padding='same')(node)   

    node = BatchNormalization()(node)
    node = Activation(activation='relu')(node)
    node = Conv2D(32, (3, 3), padding='same')(node)

    node = BatchNormalization()(node)
    node = Activation(activation='relu')(node)
    node = Conv2D(32, (3, 3), padding='same')(node)
    
    return node


def skipLayer(inp,numOut):
    
    numIn=inp.shape[3]
    if numIn==numOut:
        return inp
    else:
        return Conv2D(numOut,(1,1),padding='same')(inp)
    

def Inception_Resnet(inp,numOut):
# =============================================================================
#     Residual block pipeline 
# =============================================================================
    skip=skipLayer(inp,numOut)
    
    branch1=branch_1(inp)
    
    branch2=branch_2(inp)
    
    branch3=branch_3(inp)
    
    concat_=Concatenate()([branch1,branch2,branch3])
    conv_block=Conv2D(numOut, (1, 1), padding='same')(concat_)
    conv_block = BatchNormalization()(conv_block)
    
    return Add()([conv_block, skip])
    

def hourglass(n_levels, n_feat, inp):
# =============================================================================
#     Architecture of single Hourglass 
# =============================================================================
    
    # Upper branch
    up_branch1 = inp

    for i in range(nModules):
        up_branch1=Inception_Resnet(up_branch1,n_feat)

    # Lower branch
    low_branch1 = MaxPool2D((2, 2), strides=2)(inp)

    for i in range(nModules):
        low_branch1 = Inception_Resnet(low_branch1,n_feat)
        
    low_branch2 = None
    
    # Recursive-level Hourglass
    if n_levels > 1:
        low_branch2 = hourglass(n_levels - 1, n_feat, low_branch1)
    else:
        low_branch2 = low_branch1
        for i in range(nModules): 
            low_branch2 = Inception_Resnet(low_branch2,n_feat)

    low_branch3 = low_branch2

    for i in range(nModules):
        low_branch3 = Inception_Resnet(low_branch3,n_feat)

    up_branch2 = UpSampling2D(size=(2, 2))(low_branch3)

    # Connect two branches
    return Add()([up_branch1, up_branch2])



def gen_heatmap(inp, numOut):
# =============================================================================
#     Heatmaps generated by the left Hourglass module
# =============================================================================
    layer = Conv2D(numOut, (1,1), padding='same')(inp)
    layer = BatchNormalization()(layer)
    layer = Activation(activation='relu')(layer)
    
    return layer


def reshaper3( inn ) :
    x3d = tf.reshape( inn, [tf.shape(inn)[0],64 * 64, 16 ] )
    return x3d
    
def reshaper4( inn ) :
    x3d = tf.reshape( inn, [tf.shape(inn)[0],64 , 64, 16 ] )
    return x3d

def custom_aspp(conv):
    conv1=Conv2D(256, (1,1), padding='same')(conv)
    layer1 = BatchNormalization()(conv1)
    
    conv2=Conv2D(256, (3,3), dilation_rate=(6,6), padding='same')(conv)
    layer2 = BatchNormalization()(conv2)

    conv3=Conv2D(256, (3,3), dilation_rate=(12,12), padding='same')(conv)
    layer3 = BatchNormalization()(conv3)

    conv4=Conv2D(256, (3,3), dilation_rate=(18,18), padding='same')(conv)
    layer4 = BatchNormalization()(conv4)

    return Concatenate()([layer1, layer2, layer3, layer4])


def hg_train(njoints):
# =============================================================================
#     Stacked Hourglass Model
# =============================================================================
    global nOutChannels
    nOutChannels=njoints
    
    # Conv part for image segmentation 
    input_ = Input(shape = (None, None, 3))
    conv1=Conv2D(64, (7,7),strides=(2,2), padding='same')(input_)
    conv2=Conv2D(128, (3,3),strides=(1,1), padding='same')(conv1)
    pool=MaxPool2D((2,2),strides=2)(conv2)
    conv3=Conv2D(nFeatures, (3,3),strides=(1,1), padding='same')(pool)

    out = []
    inter = conv3
    
    # Stack multiple hourglasses end-to-end
    for i in range(nStack):
        hourglass_ = hourglass(4, nFeatures, inter)
        
        # Residual layers at output resolution
        int_step = hourglass_
        for j in range(nModules):
            int_step = Inception_Resnet(int_step,nFeatures)

        int_step = gen_heatmap(int_step, nFeatures)
        tmpOut = Conv2D(nOutChannels, (1,1), padding='same')(int_step)

        output_1a = Lambda(reshaper3) (tmpOut)

        output_1act=Softmax(axis=1)(output_1a)
        output_1thr= ThresholdedReLU(theta = 1e-4)(output_1act)
        output_1b = Lambda(reshaper4, name='stack_%d'%(i)) (output_1thr)

        
        out.append(output_1b)
        
        # Generate input for the following Hourglass
        if i < nStack-1:
            int_step_ = Conv2D(nFeatures, (1, 1), padding='same')(int_step)
            tmpOut_ = Conv2D(nFeatures, (1, 1), padding='same')(tmpOut)
            inter = Add()([inter, int_step_, tmpOut_])
    
    output_1= out[0] 
    output_2 = out[1]

    model = Model(inputs=input_, outputs=[output_1,output_2])

    return model



