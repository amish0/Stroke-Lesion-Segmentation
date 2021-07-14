# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:25:15 2020

@author: AMISH
"""

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
from mayavi import mlab
root = Tk()
root.geometry("1100x350") 
root.title("Stroke Lession Segmentation from CT and CTP data")
var=StringVar()
var_class=StringVar()
var_Data=StringVar()
var_BB=StringVar()
root.eziflag=False
root.eleflag=False
root.DataY=np.ones((16,256,256))
root.s=(16,256,256)

def displa_CT_image():
    pp=Scbar.get()
    print(int(pp))
    array=np.random.rand(256,256)*255
    array=array.astype('uint8')
    print(array)
    array=Image.fromarray(array)
    # array.save('rand.jpg')
    
    # Array1 = Image.open('rand.jpg') 
      
    # # resize the image and apply a high-quality down sampling filter 
    array = array.resize((300,200), Image.ANTIALIAS) 
      
    # # PhotoImage class is used to add image to widgets, icons etc 
    Array1 = ImageTk.PhotoImage(array) 
    input_L1=Label(root,image=Array1,borderwidth=2, relief="solid")
    input_L1.grid(row=1, column=6, padx=5,pady=5,rowspan=7,columnspan=7)
    input_L1.image=Array1
    input_L2=Label(root,image=Array1,borderwidth=2, relief="solid")
    input_L2.grid(row=1, column=13, padx=5,pady=5,rowspan=7,columnspan=7)
    input_L2.image=Array1
def display_pred_image():
    pass
def Scale_image(arg):
    displa_CT_image()
    # displa_CT_image()
    
    # display_image()
    # display_image_pred()
    
# def display_image():
#     if(hasattr(root,'data_FLAIR')):
#         temp=root.data_FLAIR
#         temp1=temp.copy()
#         temp=temp1
#         ax=int(varA.get())
#         Ho=int(varH.get())
#         Co=int(varc.get())
#         ax_im=temp[ax,:,:]
#         ax_im=(ax_im/(np.max(ax_im)+1))*255
#         ax_im=ax_im.astype('uint8')
#         ax_im=cv2.copyMakeBorder(ax_im,int((224-ax_im.shape[0])/2),
#                                  int((224-ax_im.shape[0])/2),
#                                  int((224-ax_im.shape[1])/2),
#                                  int((224-ax_im.shape[1])/2),cv2.BORDER_CONSTANT)
# #       ax_im=cv2.resize(ax_im,(224,224))
#         ax_im=Image.fromarray(ax_im)
#         ax_im.save('ax_im.ppm')
# #        cv2.imwrite('ax_im.gif',ax_im)
#         imgax = PhotoImage(file='ax_im.ppm')
#         Hx_im=temp[:,Ho,:]
#         Hx_im=(Hx_im/(np.max(Hx_im)+1))*255
#         Hx_im=Hx_im.astype('uint8')
#         Hx_im=cv2.copyMakeBorder(Hx_im,int((224-Hx_im.shape[0])/2),
#                                  int((224-Hx_im.shape[0])/2),
#                                  int((224-Hx_im.shape[1])/2),
#                                  int((224-Hx_im.shape[1])/2),cv2.BORDER_CONSTANT)
# #        Hx_im=cv2.resize(Hx_im,(224,224))
#         Hx_im=Image.fromarray(Hx_im)
#         Hx_im.save('Hx_im.ppm')
#         imgHx = PhotoImage(file='Hx_im.ppm')
#         Cx_im=temp[:,:,Co]
#         Cx_im=(Cx_im/(np.max(Cx_im)+1))*255
#         Cx_im=Cx_im.astype('uint8')
#         Cx_im=cv2.copyMakeBorder(Cx_im,int((224-Cx_im.shape[0])/2),
#                                  int((224-Cx_im.shape[0])/2),
#                                  int((224-Cx_im.shape[1])/2),
#                                  int((224-Cx_im.shape[1])/2),cv2.BORDER_CONSTANT)
# #        Cx_im=cv2.resize(Cx_im,(224,224))
#         Cx_im=Image.fromarray(Cx_im)
#         Cx_im.save('Cx_im.ppm')
# #        cv2.imwrite('Cx_im.gif', Cx_im)
#         imgCx = PhotoImage(file='Cx_im.ppm')
#         L1=Label(root,image=imgax)
#         L1.image=imgax
#         L1.grid(row=0,column=3, rowspan=6,columnspan=10)
#         L2=Label(root,image=imgHx)
#         L2.image=imgHx
#         L2.grid(row=0,column=13, rowspan=6,columnspan=10)
#         L3=Label(root,image=imgCx)
#         L3.image=imgCx
#         L3.grid(row=0,column=23, rowspan=6,columnspan=10)
        
# def display_image_pred():
#     if(hasattr(root,'DataY')):
#         temp11=root.DataY
#         temp12=temp11.copy()
#         temp11=temp12
#         s=temp11.shape
#         temp11=temp11.reshape(s[0],s[1],s[2])
#         ax1=int(varA.get())
#         Ho1=int(varH.get())
#         Co1=int(varc.get())
#         ax_im1=np.round(temp11[ax1,:,:])
#         ax_im1=(ax_im1)*255
#         ax_im1=ax_im1.astype('uint8')
#         ax_im1=cv2.copyMakeBorder(ax_im1,int((224-ax_im1.shape[0])/2),
#                                  int((224-ax_im1.shape[0])/2),
#                                  int((224-ax_im1.shape[1])/2),
#                                  int((224-ax_im1.shape[1])/2),cv2.BORDER_CONSTANT)
# #        ax_im=cv2.resize(ax_im,(224,224))
#         ax_im1=Image.fromarray(ax_im1)
#         ax_im1.save('pax_im1.ppm')
# #        cv2.imwrite('ax_im.gif',ax_im)
#         imgax1 = PhotoImage(file='pax_im1.ppm')
#         Hx_im1=temp11[:,Ho1,:]
#         Hx_im1=(Hx_im1)*255
#         Hx_im1=Hx_im1.astype('uint8')
#         Hx_im1=cv2.copyMakeBorder(Hx_im1,int((224-Hx_im1.shape[0])/2),
#                                  int((224-Hx_im1.shape[0])/2),
#                                  int((224-Hx_im1.shape[1])/2),
#                                  int((224-Hx_im1.shape[1])/2),cv2.BORDER_CONSTANT)
# #        Hx_im=cv2.resize(Hx_im,(224,224))
#         Hx_im1=Image.fromarray(Hx_im1)
#         Hx_im1.save('pHx_im1.ppm')
#         imgHx1 = PhotoImage(file='pHx_im1.ppm')
#         Cx_im1=temp11[:,:,Co1]
#         Cx_im1=(Cx_im1)*255
#         Cx_im1=Cx_im1.astype('uint8')
#         Cx_im1=cv2.copyMakeBorder(Cx_im1,int((224-Cx_im1.shape[0])/2),
#                                  int((224-Cx_im1.shape[0])/2),
#                                  int((224-Cx_im1.shape[1])/2),
#                                  int((224-Cx_im1.shape[1])/2),cv2.BORDER_CONSTANT)
# #        Cx_im=cv2.resize(Cx_im,(224,224))
#         Cx_im1=Image.fromarray(Cx_im1)
#         Cx_im1.save('pCx_im1.ppm')
# #        cv2.imwrite('Cx_im.gif', Cx_im)
#         imgCx1 = PhotoImage(file='pCx_im1.ppm')
#         L11=Label(root,image=imgax1)
#         L11.image=imgax1
#         L11.grid(row=6,column=3, rowspan=6,columnspan=10)
#         L21=Label(root,image=imgHx1)
#         L21.image=imgHx1
#         L21.grid(row=6,column=13, rowspan=6,columnspan=10)
#         L31=Label(root,image=imgCx1)
#         L31.image=imgCx1
#         L31.grid(row=6,column=23, rowspan=6,columnspan=10)


# Archi_image = Image.open('Architecture.bmp') 
      
# # resize the image and apply a high-quality down sampling filter 
# Archi_image = Archi_image.resize((300,200), Image.ANTIALIAS) 
  
# # PhotoImage class is used to add image to widgets, icons etc 
# Archi_image = ImageTk.PhotoImage(Archi_image) 
# Arci_L2=Label(root, image=Archi_image, borderwidth=2).grid(row=6, column=0, padx=5,pady=5,rowspan=7,columnspan=6)
input_L1=Label(root,text='input',height=15, width=40,borderwidth=2, relief="solid")
input_L1.grid(row=1, column=6, padx=5,pady=5,rowspan=7,columnspan=7)
# input_L1.image=Array1
input_L2=Label(root,text='predicted',height=15, width=40,borderwidth=2, relief="solid")
input_L2.grid(row=1, column=13, padx=5,pady=5,rowspan=7,columnspan=7)
# input_L2.image=Array1
Scbar = DoubleVar()
scale = Scale( root, variable = Scbar, orient=HORIZONTAL,length=500,showvalue=0,from_=0, to=root.s[0],command=Scale_image).grid(row=8, column=6, columnspan=14)

# button = Button(root, text="Get Scale Value", command=sel).grid(row=13,column=2)

# label_sc = Label(root).grid(row=14,column=2)

#BE1.focus_set()
#s=np.shape(root.data_FLAIR)
# A1=Label(root, text='Axial').grid(row=6, column=0,padx=5,pady=2,sticky=W)
# varA = DoubleVar()
    
# axial = Spinbox(root,from_ = 0, to=s[0]-1, width=30, command=Spinof_call,textvariable=varA)
# axial.grid(row=6,column=1,columnspan=3, rowspan=1,padx=5,pady=2,sticky=W)

# H1=Label(root, text='Seggital').grid(row=7, column=0,padx=5,pady=2,sticky=W)
# varH = DoubleVar()
# Hori = Spinbox(root,from_ = 0, to=s[1]-1, width=30,command=Spinof_call,textvariable=varH)
# Hori.grid(row=7,column=1,columnspan=3, rowspan=1,padx=5,pady=2,sticky=W)

# C1=Label(root, text='Corronal').grid(row=8, column=0,padx=5,pady=2,sticky=W)
# varc = DoubleVar()
# Cor = Spinbox(root,from_ = 0, to=s[2]-1, width=30,command=Spinof_call, textvariable=varc)
# Cor.grid(row=8,column=1,columnspan=3, rowspan=1,padx=5,pady=2,sticky=W)
# Button(root, text='Show Result', command=plot_maya_Brain, ).grid(row=9, column=0, columnspan=3, rowspan=1,
#             padx=5,pady=2, sticky=E+W+S+N)
# azi = DoubleVar()
# scale = Scale( root, variable = azi, from_=-180, to=180, length=400, orient = HORIZONTAL, command=view_maya ).grid(row=10,column=0,rowspan=1,columnspan=3)
# ele = DoubleVar()
# scale = Scale( root, variable = ele, from_=0, to=180, length=400,orient = HORIZONTAL, command=view_maya ).grid(row=11,column=0,rowspan=1, columnspan=3)

root.mainloop()