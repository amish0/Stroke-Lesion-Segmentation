# Function Available and Function:
#*****************************************************************************************************************************
#  rotation_finding          : Calculate the rotation and centroid of data
#  Skull_striping            : Generate the binary mask of soft tissues from mask_data(CBF+CBV+Tmax+MTT Data) and bounding box 
#  Cropping_data             : Generate an cropped data
#  Cropping_data_rev         : Reverse of cropping operation
#  Data_Normalization        : Slice wise normalization
#  patch_generation_training : Patch generation for training
#  patch_generation_forward  : left to right patch generation for prediction of training and testing dataset
#  patch_generation_sym      : right to left patch generation for prediction of training and testing dataset
#  data_recons_3D_sym        : 3D data reconstruction from predicted patches
#*****************************************************************************************************************************

import numpy as np
from skimage.morphology import reconstruction
from skimage import transform
from skimage import measure

def rotation_finding(data):
    seedd=data.max()
    # print(seedd)
    data_new=[]
    area=[]
    for i in range(data.shape[0]):
        image=(data[i]>data.min())
        seed = np.copy(image)
        seed[1:-1, 1:-1] = 1.0
        mask = image

        data_new.append(reconstruction(seed, mask, method='erosion'))
        area.append(np.sum(data_new[i]))
    area=np.stack(area)
    area_idx=np.argmax(area)
    area_data=data_new[area_idx]

    all_labels = measure.label(area_data,background=0)
    properties = measure.regionprops(all_labels)
    area=[prop.area for prop in properties]
    area=np.stack(area)
    # print(area)
    area_idx=np.argmax(area)
    y0, x0 = properties[area_idx].centroid
    orientation = properties[area_idx].orientation

    return orientation, [y0,x0]

def Skull_striping(mask_data, orient):
    region_prop=dict()
    for i in range(mask_data.shape[0]):
        image=mask_data[i]
        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.max()
        mask = image

        mask_data[i] = reconstruction(seed, mask, method='erosion')
    for i in range(mask_data.shape[0]):
        all_labels = measure.label(mask_data[i],background=0)
        properties = measure.regionprops(all_labels)
        if len(properties)==0:
            continue
        else:
            area=[prop.area for prop in properties]
            area=np.stack(area)
            area_idx=np.argmax(area)
            bbox=properties[area_idx].bbox
            filled_image=properties[area_idx].filled_image
            if filled_image.shape[0]>64 and filled_image.shape[1]>64:
                region_prop[i]={'bbox':bbox,'filled_image':filled_image, 'orientation':orient[0], 'centroid':orient[1]}
    return region_prop

def Cropping_data(data,region_prop, Normli=False, Rot=False):
    data_list=[]
    orien=0
    k=0
    for i in region_prop.keys():
        bbox=region_prop[i]['bbox']
        filled_image=region_prop[i]['filled_image']
        orien=region_prop[i]['orientation']
        centeroid=region_prop[i]['centroid']
        data_temp=data[i,bbox[0]:bbox[2],bbox[1]:bbox[3]]*filled_image
        if Rot:
            data_temp = transform.rotate(data_temp,angle=-orien*180/3.14,center=(centeroid[0]-bbox[0],centeroid[1]-bbox[1]))
            filled_image = transform.rotate(filled_image,angle=-orien*180/3.14,center=(centeroid[0]-bbox[0],centeroid[1]-bbox[1]))
        if Normli:
            filled_image=np.round(filled_image)
            filled_idx=np.argwhere(filled_image>0)
            data_temp=(data_temp-np.mean(data_temp[filled_idx[:,0],filled_idx[:,1]]))/(np.std(data_temp[filled_idx[:,0],filled_idx[:,1]])+1)
        data_list.append(data_temp)

    return data_list
	
def Cropping_data_rev(data,region_prop,SS,Rot=False):
    data_list=[]
    orien=0
    k=0
    Data1=np.zeros(SS,dtype='float32')
    for i in region_prop.keys():
        bbox=region_prop[i]['bbox']
        filled_image=region_prop[i]['filled_image']
        orien=region_prop[i]['orientation']
        centeroid=region_prop[i]['centroid']
        Data1[i,bbox[0]:bbox[2],bbox[1]:bbox[3]]=data[k]
        if Rot:
            Data1[i]=transform.rotate(Data1[i],angle=orien*180/3.14,center=(centeroid[0],centeroid[1]))
        k+=1
    return Data1

def Data_Normalization(Data):
    for j in range(len(Data)):
        m=np.mean(Data[j])
        s=np.std(Data[j])
        Data[j]=(Data[j]-m)/(s+1)
    return Data

def patch_generation_training(data_CT,data_CBF,data_Tmax,data_CBV,data_MTT,data_OT,idx,ii_jj_start=[0,4,8,12]):
    data=[]
    for i in idx:
        for j in range(len(data_CT[i])):
            temp=[data_CT[i][j],data_CBF[i][j],data_Tmax[i][j],data_CBV[i][j],data_MTT[i][j],data_OT[i][j]]
            temp=np.stack(temp,axis=-1)
            for ii_jj in ii_jj_start:
                ii=ii_jj
                inc_ii=16
                inc_jj=16
                while ii < temp.shape[0]:
                    if ii+64>=temp.shape[0]:
                        ii=temp.shape[0]-64
                        inc_ii=64
                    else:
                        inc_ii=16
                    jj=ii_jj
                    while jj < temp.shape[1]:
                        if jj+64>=temp.shape[1]:
                            jj=temp.shape[1]-64
                            inc_jj=64
                        else:
                            inc_jj=16
                        if ii_jj==0:
                            data.append(temp[ii:ii+64,jj:jj+64,:])
                            idx.append([j,ii,jj])
                        else:
                            if not ((ii == temp.shape[0]-64) or (jj == temp.shape[1]-64)):
                                data.append(temp[ii:ii+64,jj:jj+64,:])
                        jj=jj+inc_jj
                    ii=ii+inc_ii
            temp=temp[:,::-1,:]
            for ii_jj in ii_jj_start:
                ii=ii_jj
                inc_ii=16
                inc_jj=16
                while ii < temp.shape[0]:
                    if ii+64>=temp.shape[0]:
                        ii=temp.shape[0]-64
                        inc_ii=64
                    else:
                        inc_ii=16
                    jj=ii_jj
                    while jj < temp.shape[1]:
                        if jj+64>=temp.shape[1]:
                            jj=temp.shape[1]-64
                            inc_jj=64
                        else:
                            inc_jj=16
                        if ii_jj==0:
                            data.append(temp[ii:ii+64,jj:jj+64,:])
                            idx.append([j,ii,jj])
                        else:
                            if not ((ii == temp.shape[0]-64) or (jj == temp.shape[1]-64)):
                                data.append(temp[ii:ii+64,jj:jj+64,:])
                        jj=jj+inc_jj
                    ii=ii+inc_ii
    data=np.stack(data)
    OT=np.round(data[:,:,:,-1])
    OT=np.expand_dims(OT,axis=-1)
    data=data[:,:,:,:5]
    print(OT.shape)
    return data, OT

def patch_generation_forward(data_CT,data_CBF,data_Tmax,data_CBV,data_MTT,ii_jj_start=[0,4,8,12]):
    data=[]
    idx=[]
    for j in range(len(data_CT)):
        temp=[data_CT[j],data_CBF[j],data_Tmax[j],data_CBV[j],data_MTT[j]]
        temp=np.stack(temp,axis=-1)
        for ii_jj in ii_jj_start:
            ii=ii_jj
            inc_ii=16
            inc_jj=16
            while ii < temp.shape[0]:
                if ii+64>=temp.shape[0]:
                    ii=temp.shape[0]-64
                    inc_ii=64
                else:
                    inc_ii=16
                jj=ii_jj
                while jj < temp.shape[1]:
                    if jj+64>=temp.shape[1]:
                        jj=temp.shape[1]-64
                        inc_jj=64
                    else:
                        inc_jj=16
                    if ii_jj==0:
                        data.append(temp[ii:ii+64,jj:jj+64,:])
                        idx.append([j,ii,jj])
                    else:
                        if not ((ii == temp.shape[0]-64) or (jj == temp.shape[1]-64)):
                            data.append(temp[ii:ii+64,jj:jj+64,:])
                            idx.append([j,ii,jj])
                    jj=jj+inc_jj
                ii=ii+inc_ii
    data=np.stack(data)

    return data,idx

def patch_generation_sym(data_CT,data_CBF,data_Tmax,data_CBV,data_MTT,ii_jj_start=[0,4,8,12]):
    data=[]
    idx=[]
    for j in range(len(data_CT)):
        temp=[data_CT[j],data_CBF[j],data_Tmax[j],data_CBV[j],data_MTT[j]]
        temp=np.stack(temp,axis=-1)
        temp=temp[:,::-1,:]
        for ii_jj in ii_jj_start:
            ii=ii_jj
            inc_ii=16
            inc_jj=16
            while ii < temp.shape[0]:
                if ii+64>=temp.shape[0]:
                    ii=temp.shape[0]-64
                    inc_ii=64
                else:
                    inc_ii=16
                jj=ii_jj
                while jj < temp.shape[1]:
                    if jj+64>=temp.shape[1]:
                        jj=temp.shape[1]-64
                        inc_jj=64
                    else:
                        inc_jj=16
                    if ii_jj==0:
                        data.append(temp[ii:ii+64,jj:jj+64,:])
                        idx.append([j,ii,jj])
                    else:
                        if not ((ii == temp.shape[0]-64) or (jj == temp.shape[1]-64)):
                            data.append(temp[ii:ii+64,jj:jj+64,:])
                            idx.append([j,ii,jj])
                    jj=jj+inc_jj
                ii=ii+inc_ii
    data=np.stack(data)

    return data,idx

def data_recons_3D_sym(Data,idx,Data_shape,Data_rev,idx_rev):
    result=[]
    tt1=[]
    idx=np.stack(idx)
    p=idx[:,0]
    p1=np.unique(p)
    for i in p1:
        p2=np.argwhere(idx[:,0]==i)
        S=Data_shape[i].shape
        tt=np.zeros(S,dtype='float32')
        tt_one=np.zeros(S,dtype='float32')
        for j in p2:
            idx_temp=idx[j[0]]
            # print(idx_temp)
            tt[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]=tt[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]+Data[j[0]]
            tt_one[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]=tt_one[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]+np.ones((64,64),dtype='float32')
        tt1.append(tt_one)
        result.append(tt)

    result_rev=[]
    tt2=[]
    idx_rev=np.stack(idx_rev)
    p=idx_rev[:,0]
    p1=np.unique(p)
    for i in p1:
        p2=np.argwhere(idx_rev[:,0]==i)
        S=Data_shape[i].shape
        tt=np.zeros(S,dtype='float32')
        tt_one=np.zeros(S,dtype='float32')
        for j in p2:
            idx_temp=idx_rev[j[0]]
            tt[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]=tt[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]+Data_rev[j[0]]
            tt_one[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]=tt_one[idx_temp[1]:idx_temp[1]+64,idx_temp[2]:idx_temp[2]+64]+np.ones((64,64),dtype='float32')
        tt=tt[:,::-1]
        tt_one=tt_one[:,::-1]
        tt2.append(tt_one)
        result_rev.append(tt)
    result1=[]
    for i in range(len(result)):
        resultc=result[i]+result_rev[i]
        tt_one_c=tt1[i]+tt2[i]
        resultc=np.round(resultc/tt_one_c)
        result1.append(resultc)
    return result1 #res_one
