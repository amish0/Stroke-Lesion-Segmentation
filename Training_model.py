# main architecture defined for proposed work: unet_am

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from metrics_and_loss import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Add
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation

def Encoder_block(input_en, filter_no, Droup_out=False):
    concat_up0 = Conv2D(filter_no, (3,3), padding = 'same', kernel_initializer= 'he_normal')(input_en)
    concat_up0 = BatchNormalization()(concat_up0)
    concat_up0 = Activation('relu')(concat_up0)
    concat_up0 = Conv2D(filter_no, (3,3), padding = 'same', kernel_initializer= 'he_normal')(concat_up0)
    concat_up0 = BatchNormalization()(concat_up0)
    concat_up0 = Activation('relu')(concat_up0)
    concat_up1 = Conv2D(filter_no, (1,1), padding = 'same', kernel_initializer= 'he_normal')(input_en)
    concat_up1 = BatchNormalization()(concat_up1)
    concat_up1 = Activation('relu')(concat_up1)
    concat_up1=Add()([concat_up1,concat_up0])
    return concat_up1

def Decoder_block(input_de, concat_De, filter_no,up_size=(2,2)):
    De_up = Conv2D(filter_no,2, padding = 'same',
                 kernel_initializer = 'he_normal')(UpSampling2D(size = up_size)(input_de))
    De_merge = concatenate([concat_De, De_up], axis = -1)
    De_merge1 = Conv2D(filter_no, (3,3), padding = 'same', kernel_initializer= 'he_normal')(De_merge)
    De_merge1 = BatchNormalization()(De_merge1)
    De_merge1 = Activation('relu')(De_merge1)
    res_x=Conv2D(filter_no,(1,1), padding = 'same', kernel_initializer= 'he_normal')(De_merge)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)
    De_merge1=Add()([De_merge1,res_x])
    return De_merge1

def unet_am(input_size = (64,64,5)):
    input1 = Input(input_size, name = 'input_t1')
    En1  = Encoder_block(input1, filter_no=8)

    En_pool = MaxPooling2D(pool_size = (2,2), padding="same")(En1)

    En2 = Encoder_block(En_pool, filter_no=16)

    En_poo2 = MaxPooling2D(pool_size = (2,2),padding="same")(En2)

    En3 = Encoder_block(En_poo2, filter_no=32)

    En_poo3 = MaxPooling2D(pool_size = (2,2),padding="same")(En3)

    B_neck0 = Conv2D(64, (3,3), padding = 'same', kernel_initializer= 'he_normal')(En_poo3)
    B_neck0 = BatchNormalization()(B_neck0)
    B_neck0 = Activation('relu')(B_neck0)
    
    B_neck1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer= 'he_normal')(B_neck0)
    B_neck1 = BatchNormalization()(B_neck1)
    B_neck1 = Activation('relu')(B_neck1)

    De3 = Decoder_block(B_neck1, En3, filter_no=8,up_size=(2,2))
    De2 = Decoder_block(De3, En2, filter_no=8,up_size=(2,2))
    De1 = Decoder_block(De2, En1, filter_no=8) 

    out = Conv2D(1,(1,1),padding='same', activation = 'sigmoid', name='result1')(De1)

    model = Model(inputs = input1, outputs = out)

    model.compile(optimizer = Adam(lr = 1e-2), loss = tversky_loss, metrics = ['accuracy',dice_coeff,precision,dice_loss,recall])
    
    return model
