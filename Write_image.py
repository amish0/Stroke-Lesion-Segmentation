import nibabel as nib
import numpy as np
import cv2
import os

Path='Result/Training' # path to '.nii' predicted result
Dest_path='Result_image/Training' # Destination path for writing image
p=os.listdir(Path)

for i in p:
    x=nib.load(os.path.join(Path,i))
    img=x.get_fdata()
    img=img.astype('float32')
    img=img*255
    img=img.astype('uint8')
    img=img.T
    # print(np.unique(img))
    if not os.path.isdir(os.path.join(Dest_path,i[:-4])):
        os.makedirs(os.path.join(Dest_path,i[:-4]))
    else:
        print('Directory exits')
    for j in range(img.shape[0]):
        cv2.imwrite(os.path.join(Dest_path,i[:-4],str(j)+'.png'),img[j])