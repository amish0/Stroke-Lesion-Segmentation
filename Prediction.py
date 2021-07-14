import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
import os
import glob
import nibabel as nib
from Training_model import unet_am
from pre_processing import *

# Testing Prediction
# Dataset Path for training/testining data
# for prediction on training change the "dataset path" to training data directory
# and Result_path to 'Result\\Training/'

Dataset='ISLES2018_Testing/TESTING/' 
Result_path='Result\\Testing/'
pp=os.listdir(Dataset)
print(pp)
YCBF=[glob.glob(Dataset+q1+('/*.CT_CBF*/*.nii'))[0] for q1 in pp[:]]
YCBV=[glob.glob(Dataset+q1+('/*.CT_CBV*/*.nii'))[0] for q1 in pp[:]]
YMTT=[glob.glob(Dataset+q1+('/*.CT_MTT*/*.nii'))[0] for q1 in pp[:]]
YTmax=[glob.glob(Dataset+q1+('/*.CT_Tmax*/*.nii'))[0] for q1 in pp[:]]
YCT=[glob.glob(Dataset+q1+('/*.CT.*/*.nii'))[0] for q1 in pp[:]]

# Model 
model=unet_am() # defined model

# Dataset generation and Prediction

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
result=[]
for i in range(len(YCT)):
    x=nib.load(YCT[i])
    img=x.get_fdata()
    img=img.astype('float32').T
    Data4=img
    SSS=img.shape
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

    temp=Data1+Data2+Data3+Data0
    
    temp1=(temp>0)*1.0 # Non zeros mask creation
    
    rot=rotation_finding(Data4) 
    region_prop=Skull_striping(temp1,rot) # 

    Temp_CT=Cropping_data(Data4,region_prop,Normli=False,Rot=False)
    Temp_CT=Data_Normalization(Temp_CT)
    Temp_CBF=Cropping_data(Data0,region_prop,Normli=False,Rot=False)
    Temp_CBF=Data_Normalization(Temp_CBF)

    Temp_Tmax=Cropping_data(Data1,region_prop,Normli=False,Rot=False)
    Temp_Tmax=Data_Normalization(Temp_Tmax)

    Temp_CBV=Cropping_data(Data2,region_prop,Normli=False,Rot=False)
    Temp_CBV=Data_Normalization(Temp_CBV)

    Temp_MTT=Cropping_data(Data3,region_prop,Normli=False,Rot=False)
    Temp_MTT=Data_Normalization(Temp_MTT)

    print(['idx',i])
    
    # Patch generation from data
    
    patch,patch_idx=patch_generation_forward(Temp_CT,Temp_CBF,Temp_Tmax,Temp_CBV,Temp_MTT)
    patch_rev,patch_idx_rev=patch_generation_sym(Temp_CT,Temp_CBF,Temp_Tmax,Temp_CBV,Temp_MTT)
    
    # Prediction
    
    res=np.zeros(patch.shape[:-1])
    res2=np.zeros(patch_rev.shape[:-1])
    for j in range(5):
        model.load_weights('Module_Weight\\GT_12_10_2020'+str(j)+'_unet.h5')
        temp_res=np.squeeze(model.predict(patch))
        temp_res=np.round(temp_res)
        res=res+temp_res
    res=res/4.8
    res=np.round(res)

    for j in range(5):
        model.load_weights('Module_weight\\GT_12_10_2020'+str(j)+'_unet.h5')
        temp_res=np.squeeze(model.predict(patch_rev))
        temp_res=np.round(temp_res)
        res2=res2+temp_res
    res2=res2/4.8
    res2=np.round(res2)
    
    # Dataset reconstruction from predicted pateches
    res1 = data_recons_3D_sym(res,patch_idx,Temp_CT,res2,patch_idx_rev)

    res1=Cropping_data_rev(res1,region_prop,SSS,Rot=False)
    
    # Savining the predicted data in ".nii" format
    
    res1=np.round(np.stack(res1))
    res1=res1.astype(int)
    res1=res1.T
    x=nib.load(YMTT[i])
    img=x.get_fdata()
    hed=x.header

    print(np.shape(res1))
    print(np.shape(img))    
    clipped_img = nib.Nifti1Image(res1, x.affine, x.header)
    nib.save(clipped_img, Result_path+'VSD.my_result_01'+'.'+YMTT[i][-10:-4]+'.nii')