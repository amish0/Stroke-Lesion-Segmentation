import numpy as np
import nibabel as nib
import os
import glob
import matplotlib.pyplot as plt
import h5py
from skimage import measure
import random
from pre_processing import *

Dataset='ISLES2018_Training\\TRAINING/'
pp=os.listdir(Dataset)
print(pp)
YCBF=[glob.glob(Dataset+q1+('/*.CT_CBF*/*.nii'))[0] for q1 in pp[:]]
YCBV=[glob.glob(Dataset+q1+('/*.CT_CBV*/*.nii'))[0] for q1 in pp[:]]
YMTT=[glob.glob(Dataset+q1+('/*.CT_MTT*/*.nii'))[0] for q1 in pp[:]]
YTmax=[glob.glob(Dataset+q1+('/*.CT_Tmax*/*.nii'))[0] for q1 in pp[:]]
YCT=[glob.glob(Dataset+q1+('/*.CT.*/*.nii'))[0] for q1 in pp[:]]
YOT=[glob.glob(Dataset+q1+('/*.OT.*/*.nii'))[0] for q1 in pp[:]]
# F0, F1, F2, F3, F4--> case no assigned to each fold
F0=[59,14,89,13,25,38,60,70,94,46,53,28,40,76,47,72,10,67,34]
F1=[11,44,43,66,57,48,75,79,51,69,5,12,93,52,74,26,1,84]
F2=[56,49,23,19,92,50,58,39,62,82,35,63,21,16,31,15,8,91,33]
F3=[83,77,32,73,22,87,29,41,61,78,18,30,9,90,36,24,65,4,81]
F4=[2,71,6,85,45,20,17,80,7,64,68,37,42,86,27,88,55,3,54]
F0_train,F1_train,F2_train,F3_train,F4_train=[],[],[],[],[]
F0_val,F1_val,F2_val,F3_val,F4_val=[],[],[],[],[]

Data_CT_OT=[]
Data_CT=[]
Data_OT=[]
Data_CBV=[]
Data_CBF=[]
Data_MTT=[]
Data_Tmax=[]
mask=[]
patch_size=64
patch_idx=[]
for i in range(len(YCBF)):
    print(i)
    x=nib.load(YCT[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    Data4=img

    # d=x.header
    x=nib.load(YCBF[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    Data0=img

    x=nib.load(YTmax[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    Data1=img

    x=nib.load(YCBV[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    Data2=img

    x=nib.load(YMTT[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    Data3=img

    x=nib.load(YOT[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    temp_OT=img

    temp=Data1+Data2+Data3+Data0
    temp1=(temp>0)*1.0
    rot=rotation_finding(Data4)
    region_prop=Skull_striping(temp1,rot)

    Data4=Cropping_data(Data4,region_prop)
    Data_CT.append(Data_Normalization(Data4))

    Data0=Cropping_data(Data0,region_prop)
    Data_CBF.append(Data_Normalization(Data0))

    Data1=Cropping_data(Data1,region_prop)
    Data_Tmax.append(Data_Normalization(Data1))

    Data2=Cropping_data(Data2,region_prop)
    Data_CBV.append(Data_Normalization(Data2))

    Data3=Cropping_data(Data3,region_prop)
    Data_MTT.append(Data_Normalization(Data3))

    temp_OT=Cropping_data(temp_OT,region_prop)
    Data_OT.append(temp_OT)
    if int(pp[i][5:]) in F0:
        F0_val.append(i)
    else:
        F0_train.append(i)
    if int(pp[i][5:]) in F1:
        F1_val.append(i)
    else:
        F1_train.append(i)
    if int(pp[i][5:]) in F2:
        F2_val.append(i)
    else:
        F2_train.append(i)
    if int(pp[i][5:]) in F3:
        F3_val.append(i)
    else:
        F3_train.append(i)
    if int(pp[i][5:]) in F4:
        F4_val.append(i)
    else:
        F4_train.append(i)

F_idx={0:{'train':F0_train,'val':F0_val},
       1:{'train':F1_train,'val':F1_val},
       2:{'train':F2_train,'val':F2_val},
       3:{'train':F3_train,'val':F3_val},
       4:{'train':F4_train,'val':F4_val}}

for i, val in F_idx.items():
    train_Data,train_OT,val_Data,val_OT=[],[],[],[]
    train_Data, train_OT=patch_generation_fine_tunning(Data_CT,Data_CBF,Data_Tmax,Data_CBV,Data_MTT,Data_OT,val['train'],ii_jj_start=[0,4,8,12])
    val_Data, val_OT=patch_generation_fine_tunning(Data_CT,Data_CBF,Data_Tmax,Data_CBV,Data_MTT,Data_OT,val['val'],ii_jj_start=[0,4,8,12])
    hf=h5py.File("GT_Whole_RN16_ISLES2018_F"+str(i)+".hdf5", "w")
    hf.create_dataset('train_Data',data=train_Data)
    hf.create_dataset('train_OT',data=train_OT)
    hf.create_dataset('val_Data',data=val_Data)
    hf.create_dataset('val_OT',data=val_OT)
    hf.close()
    
    