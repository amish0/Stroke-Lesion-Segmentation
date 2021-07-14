
from tkinter import *
#root = Tk()
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import tkinter
import cv2
import nibabel as nib
import tensorflow as tf
from metrics_and_loss import *
# from fractal_net import fractunet
from Training_model import unet_am
from pre_processing import *
from skimage import measure
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import ImageTk, Image

#from xlwt import Workbook 
import matplotlib.pyplot as plt
import glob
from mayavi import mlab
root = Tk()
root.geometry("1200x450") 
root.title("Stroke Lession Segmentation from CT and CTP data")
var=StringVar()
var_class=StringVar()
var_Data=StringVar()
var_BB=StringVar()
root.eziflag=False
root.eleflag=False
root.DataY=np.ones((16,256,256))
root.s=(16,256,256)
model=unet_am((64,64,5))
model.compile(optimizer='Adam', loss=bce_dice_loss, metrics=[recall, precision, specificity, mean_iou, dice_coeff,dice_loss,'accuracy'])
root.Dataload=False
root.Prediction=False
root.skull=False
root.overlay=False
root.scale_flag=False
    
def load_file():
    input_L1=Label(root,text='input',height=18, width=45,borderwidth=2, relief="solid")
    input_L1.grid(row=1, column=6, padx=5,pady=5,rowspan=8,columnspan=7)
    # input_L1.image=Array1
    input_L2=Label(root,text='predicted',height=18, width=45,borderwidth=2, relief="solid")
    input_L2.grid(row=1, column=13, padx=5,pady=5,rowspan=8,columnspan=7)
    Scale_Visualization(state_scale='DISABLE')
        
    def read_path():
        root.Path = filedialog.askdirectory()
        # filedialog.askopenfilename(initialdir = "D:/work/amish/stroke_lession/SISS2015_Training/2",
                                                # title = "Load File",filetypes = (("NEFT TYPE","*.nii"),("all files","*.*")))
        var.set(root.Path)
        var_Path.set(root.Path)
        print(root.Path)
 
    def quit_windows():
        top.destroy()
        if hasattr(root,'CT'):
            delattr(root,'CT')
        if hasattr(root,'CBF'):
            delattr(root,'CBF')
        if hasattr(root,'CBV'):
            delattr(root,'CBV')
        if hasattr(root,'MTT'):
            delattr(root,'MTT')
        if hasattr(root,'Tmax'):
            delattr(root,'Tmax')
        if hasattr(root,'Path'):
            delattr(root,'Path')
        var.set([])
    def OK_windows():
        top.destroy()
#        k=k+1
    top = Toplevel(root)
    var_CT=StringVar(top)
    var_CBF=StringVar(top)
    var_CBV=StringVar(top)
    var_MTT=StringVar(top)
    var_Tmax=StringVar(top)
    var_Path=StringVar(top)
    top.title("Load CT and CTP Data")
    top.maxsize(400,500)
    top.minsize(400,200)
    
    Label(top, text='Path').grid(row=0, column=0,padx=5,pady=2,sticky=W)
    Entry(top,textvariable=var_CT,selectborderwidth=2, width=30).grid(row=0,
    column=1,columnspan=2,padx=5,pady=2,sticky=W)
    Button(top, text='Browse', command=read_path,width=10).grid(row=0, column=3, columnspan=1, rowspan=1)

    Button(top, text='OK', command=OK_windows,width=13).grid(row=5, column=1, columnspan=1, rowspan=1)
    Button(top, text='Cancel', command=quit_windows,width=13).grid(row=5, column=2, columnspan=1, rowspan=1,sticky=W)
    top.mainloop() 

                    
def read_file():
    if(hasattr(root,'Path')):
        root.CBF=glob.glob(root.Path+('/*.CT_CBF*/*.nii'))[0]
        root.CBV=glob.glob(root.Path+('/*.CT_CBV*/*.nii'))[0]
        root.MTT=glob.glob(root.Path+('/*.CT_MTT*/*.nii'))[0]
        root.Tmax=glob.glob(root.Path+('/*.CT_Tmax*/*.nii'))[0]
        root.CT=glob.glob(root.Path+('/*.CT.*/*.nii'))[0]
        # YOT=glob.glob(Dataset+('/*.OT.*/*.nii'))[0]
        x=nib.load(root.CT)
        img=x.get_fdata()
        img=img.astype('float32').T
        root.CT=img
        root.s=img.shape
    #     # d=x.header
        x=nib.load(root.CBF)
        img=x.get_fdata()
        img=img.astype('float32').T
        root.CBF=img
    
        x=nib.load(root.Tmax)
        img=x.get_fdata()
        img=img.astype('float32').T
        root.Tmax=img
    
        x=nib.load(root.CBV)
        img=x.get_fdata()
        img=img.astype('float32').T
        root.CBV=img
    
        x=nib.load(root.MTT)
        img=x.get_fdata()
        img=img.astype('float32').T
        root.MTT=img
        root.Dataload=True
        root.Prediction=False
        print('Data Loading Completed')
        # Scale_Visualization()
        Scale_Visualization(state_scale='TRUE')

def preprocessing_Data():
    
    temp=root.CBV+root.CBF+root.MTT+root.Tmax
    
    temp1=(temp>0)*1.0 # Non zeros mask creation
    
    rot=rotation_finding(root.CT) 
    region_prop=Skull_striping(temp1,rot) # 

    Temp_CT=Cropping_data(root.CT,region_prop,Normli=False,Rot=False)
    Temp_CT1=Cropping_data(root.CT,region_prop,Normli=False,Rot=False)
    Temp_CT=Data_Normalization(Temp_CT)
    Temp_CBF=Cropping_data(root.CBF,region_prop,Normli=False,Rot=False)
    Temp_CBF=Data_Normalization(Temp_CBF)

    Temp_Tmax=Cropping_data(root.Tmax,region_prop,Normli=False,Rot=False)
    Temp_Tmax=Data_Normalization(Temp_Tmax)

    Temp_CBV=Cropping_data(root.CBV,region_prop,Normli=False,Rot=False)
    Temp_CBV=Data_Normalization(Temp_CBV)

    Temp_MTT=Cropping_data(root.MTT,region_prop,Normli=False,Rot=False)
    Temp_MTT=Data_Normalization(Temp_MTT)
    
    # Patch generation from data
    
    patch,patch_idx=patch_generation_forward(Temp_CT,Temp_CBF,Temp_Tmax,Temp_CBV,Temp_MTT)
    patch_rev,patch_idx_rev=patch_generation_sym(Temp_CT,Temp_CBF,Temp_Tmax,Temp_CBV,Temp_MTT)

    root.patch=patch
    root.patch_rev=patch_rev
    root.patch_idx=patch_idx
    root.patch_idx_rev=patch_idx_rev
    root.region_prop=region_prop
    root.Temp_CT=Temp_CT
    root.Temp_CT1=Cropping_data_rev(Temp_CT1,root.region_prop,root.s,Rot=False)
    
    del Temp_MTT, Temp_CBF, Temp_CBV, Temp_Tmax, temp, temp1, region_prop,rot, patch,patch_idx,patch_rev,patch_idx_rev
    print('pre Processing completed')
    
def Predict_Data():
    
    res=np.zeros(root.patch.shape[:-1])
    res2=np.zeros(root.patch_rev.shape[:-1])
    for j in range(5):
        model.load_weights('Module_Weight\\GT_12_10_2020'+str(j)+'_unet.h5')
        temp_res=np.squeeze(model.predict(root.patch))
        temp_res=np.round(temp_res)
        res=res+temp_res
    res=res/4.8
    res=np.round(res)
    print('prediction left to right patch completed')
    for j in range(5):
        model.load_weights('Module_weight\\GT_12_10_2020'+str(j)+'_unet.h5')
        temp_res=np.squeeze(model.predict(root.patch_rev))
        temp_res=np.round(temp_res)
        res2=res2+temp_res
    res2=res2/4.8
    res2=np.round(res2)
    print('prediction Right to left patch completed')
    # Dataset reconstruction from predicted pateches
    res1 = data_recons_3D_sym(res,root.patch_idx,root.Temp_CT,res2,root.patch_idx_rev)
    print('3D reconstruction completed')
    res1=Cropping_data_rev(res1,root.region_prop,root.s,Rot=False)
    print('Cropping reverse operation completed')
    root.Prediction=True
    Scale_Visualization()
    res1=res1*255
    res1=res1.astype('uint8')
    root.DataY=res1    
    temp=root.DataY
    s1=temp.shape
    print(s1)
    temp=temp.reshape(s1[0],s1[1],s1[2])
    temp=np.round(temp)
    pp=np.argwhere(temp>0)
    text='['+str(np.min(pp[:,0]))+','+str(np.min(pp[:,1]))+','+str(np.min(pp[:,2]))+','+str(np.max(pp[:,0])-np.min(pp[:,0]))+','+str(np.max(pp[:,1])-np.min(pp[:,1]))+','+str(np.max(pp[:,2])-np.min(pp[:,2]))+']'
    var_BB.set(text)
    Scale_Visualization(state_scale='True')
    # display_image_pred()
    print('pridiction completed')

def displa_CT_image():
    pp=Scbar.get()
    if root.Dataload:
        CT=np.clip(root.CT,a_min=0, a_max=110)
        CT=CT/110
        array=CT[int(pp)]*255
        # print(int(pp))
        # array=np.random.rand(256,256)*255
        array=array.astype('uint8')
        # print(array)
        array=Image.fromarray(array)
        # array.save('rand.jpg')
        
        # Array1 = Image.open('rand.jpg') 
          
        # # resize the image and apply a high-quality down sampling filter 
        array = array.resize((358,360), Image.ANTIALIAS) 
          
        # # PhotoImage class is used to add image to widgets, icons etc 
        Array1 = ImageTk.PhotoImage(array) 
        input_L1=Label(root,image=Array1,borderwidth=2, relief="solid")
        input_L1.grid(row=1, column=6, padx=5,pady=5,rowspan=8,columnspan=7)
        input_L1.image=Array1
    if root.Prediction:
        # print([pp,root.Prediction, root.skull, root.overlay])
        if root.skull and root.overlay:
            CT1=root.Temp_CT1[int(pp)]
            CT1=np.clip(CT1,a_min=0, a_max=110)
            CT1=CT1/110
            temp_pred=root.DataY[int(pp)]
            temp_idx=np.argwhere(temp_pred>0)
            if len(temp_idx)>0:
                array_pred=np.zeros((CT1.shape[0],CT1.shape[1],3))
                array_pred[:,:,0]=CT1
                array_pred[:,:,1]=CT1
                array_pred[:,:,2]=CT1
                array_pred[temp_idx[:,0],temp_idx[:,1],2]=0
                array_pred[temp_idx[:,0],temp_idx[:,1],1]=0
            else:
                array_pred=CT1
            array_pred=array_pred*255
            array_pred=array_pred.astype('uint8')
            array_pred=Image.fromarray(array_pred)
            array_pred = array_pred.resize((358,360), Image.ANTIALIAS) 
            
        elif root.skull and not root.overlay:
            CT1=root.Temp_CT1[int(pp)]
            CT1=np.clip(CT1,a_min=0, a_max=110)
            CT1=CT1/110
            temp_pred=root.DataY[int(pp)]
            temp_idx=np.argwhere(temp_pred>128)
            array_pred=CT1
            if len(temp_idx)>0:
                array_pred[temp_idx[:,0],temp_idx[:,1]]=1
            array_pred=array_pred*255
            array_pred=array_pred.astype('uint8')
            array_pred=Image.fromarray(array_pred)
            array_pred = array_pred.resize((358,360), Image.ANTIALIAS) 
            
        elif not root.skull and root.overlay:
            temp_pred=root.DataY[int(pp)]
            # temp_idx=np.argwhere(temp_pred>0)
            array_pred=np.zeros((temp_pred.shape[0],temp_pred.shape[1],3))
            array_pred[:,:,0]=temp_pred
            # array_pred=array_pred
            array_pred=array_pred.astype('uint8')
            array_pred=Image.fromarray(array_pred)
            array_pred = array_pred.resize((358,360), Image.ANTIALIAS)

        else:
            array_pred=root.DataY[int(pp)]
            # print([np.min(array_pred),np.max(array_pred)])
            # array_pred=array_pred*255
            array_pred=array_pred.astype('uint8')
            array_pred=Image.fromarray(array_pred)
            array_pred = array_pred.resize((358,360), Image.ANTIALIAS) 

        Array1_pred = ImageTk.PhotoImage(array_pred) 
        
        input_L2=Label(root,image=Array1_pred,borderwidth=2, relief="solid")
        input_L2.grid(row=1, column=13, padx=5,pady=5,rowspan=8,columnspan=7)
        input_L2.image=Array1_pred

def display_pred_image():
    pass
def Scale_image(arg):
    displa_CT_image()
Scbar = DoubleVar()
def Scale_Visualization(state_scale='DISABLE'):
    if state_scale=='DISABLE':
        scale = Scale(root, variable = Scbar, orient=HORIZONTAL,length=500,showvalue=0,from_=0, to=root.s[0]-1,command=Scale_image,state='disable').grid(row=9, column=6, columnspan=14)
    else:
        scale = Scale(root, variable = Scbar, orient=HORIZONTAL,length=500,showvalue=0,from_=0, to=root.s[0]-1,command=Scale_image).grid(row=9, column=6, columnspan=14)

def Skul_radio():
    skull_value=Skull_var.get()
    if skull_value==1:
        root.skull=True
    else:
        root.skull=False
def Overlay_radio():
    Overlay_value=Overlay_var.get()
    if Overlay_value==1:
        root.overlay=True
    else:
        root.overlay=False

LoadData=Button(root, text='Load File', command=load_file)

LoadData.grid(row=1, column=0, columnspan=1, rowspan=1,
            padx=5,pady=2, sticky=E+W+S+N)
Button(root, text='upload', command=read_file).grid(row=1, column=1, columnspan=3, rowspan=1,
            padx=5,pady=2, sticky=E+W+S+N)
Button(root, text='Preprocessing', command=preprocessing_Data).grid(row=2, column=0, columnspan=1, rowspan=1,
            padx=5,pady=2, sticky=E+W+S+N)

Button(root, text='Predict', command=Predict_Data).grid(row=2, column=1, columnspan=3, rowspan=1,
            padx=5,pady=2, sticky=E+W+S+N)

Skull_label=Label(root, text='Skull Removal').grid(row=3, column=0,padx=5,pady=2,sticky=W)

Skull_var = IntVar()
Skull_var.set(2)
R1 = Radiobutton(root, text="Yes", variable=Skull_var, value=1, command=Skul_radio).grid(row=3, column=1,padx=5,pady=2,sticky=W)

R2 = Radiobutton(root, text="No", variable=Skull_var, value=2, command=Skul_radio).grid(row=3, column=2,padx=0,pady=2,sticky=W)

Overlay_label=Label(root, text='Region Color').grid(row=4, column=0,padx=5,pady=2,sticky=W)
Overlay_var = IntVar()
Overlay_var.set(2)
R3 = Radiobutton(root, text="Yes", variable=Overlay_var, value=1, command=Overlay_radio).grid(row=4, column=1,padx=5,pady=2,sticky=W)

R4 = Radiobutton(root, text="No", variable=Overlay_var, value=2, command=Overlay_radio).grid(row=4, column=2,padx=0,pady=2,sticky=W)

L1=Label(root, text='Path').grid(row=5, column=0,padx=5,pady=2,sticky=W)

E1=Entry(root,textvariable=var,selectborderwidth=2, width=30).grid(row=5,
        column=1,columnspan=3,padx=5,pady=2,sticky=W)

B1=Label(root, text='Bounding Box').grid(row=7, column=0,padx=5,pady=2,sticky=W)

BE1=Entry(root,textvariable=var_BB,selectborderwidth=2, width=30).grid(row=7,
         column=1,columnspan=3, rowspan=1,padx=5,pady=2,sticky=W)

Archi_image = Image.open('Architecture.bmp')
      
# resize the image and apply a high-quality down sampling filter 
Archi_image = Archi_image.resize((300,200), Image.ANTIALIAS) 
  
# PhotoImage class is used to add image to widgets, icons etc 
Archi_image = ImageTk.PhotoImage(Archi_image) 
Arci_L2=Label(root, image=Archi_image, borderwidth=2).grid(row=8, column=0, padx=5,pady=5,rowspan=7,columnspan=6)
input_L1=Label(root,text='input',height=18, width=45,borderwidth=2, relief="solid")
input_L1.grid(row=1, column=6, padx=5,pady=5,rowspan=8,columnspan=7)
# input_L1.image=Array1
input_L2=Label(root,text='predicted',height=18, width=45,borderwidth=2, relief="solid")
input_L2.grid(row=1, column=13, padx=5,pady=5,rowspan=8,columnspan=7)

root.mainloop()