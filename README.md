# Stroke-Lesion-Segmentation
A lightweight asymmetric U-Net based framework to leverage acute ischemic stroke lesion segmentation in CT and CTP images
"The Presented work is trained as well as tested on ISLES2018 challenge dataset"

Dataset:
     ISLES2018 Challange Dataset (https://www.smir.ch/ISLES/Start2018)

## Requirement:
	System:      Graphics Enable
  
	Environment: Anaconda--> Spyder(Python3.8)
  
	Library:     1. Tensorflow 2.3
		     2. Tensorboard 2.3
		     3. numpy 1.18.5
		     4. skimage 0.16.2
		     5. h5py 2.10.0
		     6. glob 0.7

## File Description:
	1. Training_model.py:   Proposed model file
		     	        Called By: Prediction.py
                                	   Training.py                             
	2. pre_processinig.py:  Function required fro pre_processing the data before training and prediction
		      	        Called By: Prediction.py
	                                   Training_Data.py
    3.Training_Data.py:     It will generate training dataset From ISLES2018 Training dataset(Change Line
			                 no 11 accordining to training data directory) and save it as:-
			                 1. "GT_Whole_RN16_ISLES2018_F0.hdf5"--> Fold0
			                 2. "GT_Whole_RN16_ISLES2018_F1.hdf5"--> Fold1
			                 3. "GT_Whole_RN16_ISLES2018_F2.hdf5"--> Fold2
			                 4. "GT_Whole_RN16_ISLES2018_F3.hdf5"--> Fold3
			                 5. "GT_Whole_RN16_ISLES2018_F4.hdf5"--> Fold4
    4. Training.py:	        For training, Training Weight will be saved in folder "Module_Weight" folder for each fold.
    5. Prediction.py:       For prediction: change 
                                     line no: 16 for Training/Testing Dataset path
                                     line no: 17 for Destination path of predicted data on Training/Testining.

** Trained weight is available in folder "pre_trained_weight" and "Module_Weight" **

** Download the Dataset from the link provided in Dataset part by completing the registration process and place it in the current directory" **



       
	
