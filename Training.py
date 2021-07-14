import h5py
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from Training_model import *
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard

# Training of the model
for i in range(5):
    hf = h5py.File('GT_Whole_RN16_ISLES2018_F'+str(i)+'.hdf5', 'r') # load the dataset for training
    print(list(hf.keys()))
    x_t=hf['train_Data'][:]
    x_g=hf['train_OT'][:]
    v_t=hf['val_Data'][:]
    v_g=hf['val_OT'][:]
    hf.close()
    model=unet_am()
    Name='GT_12_10_2020'
    tensorboard = TensorBoard(log_dir="./tensor_board/logs/{}".format(Name+str(i)+'_unet'))
    MC = ModelCheckpoint(Name+str(i)+'_unet.h5', monitor='val_loss', verbose=1, save_best_only=True)
    Es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2)
    model.fit(x=x_t,y=x_g, batch_size=32,epochs=200, callbacks=[MC, Es,rlr, tensorboard],shuffle=True, validation_data=(v_t,v_g))