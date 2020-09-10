# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:33:29 2020

@author: Steven
"""


import matplotlib.pyplot as plt

import numpy as np

from numpy import genfromtxt

import pydicom

import os

import gdcm

import pickle

from skimage.transform import resize
    
import pandas as pd

import random


# The following is a useful command to view an image that is stored as a
# pixel array
# plt.imshow(  pixelarray  , cmap=plt.cm.bone) 




# dataPath is the path where the data is stored. 

dataPath = 'D:\\Kaggle\\train'



# createPath concatenates root, Dir and file into a full path name 

def createPath(Dir, subDir, file):
    return Dir + '\\' + subDir + '\\' + file

# remdcm removes .dcm from the end of a string, then converts it to an integer.

def remdcm(x):
    return int(x.strip('.dcm'))  
 
# adddcm is the inverse of remdcm. It converts x to a string, then adds .dcm 
# to the end    

def adddcm(x):
    return  str(x) + '.dcm'

# Save saves Object as filename.pkl in the working directory

def Save(Object, filename):
    
    file = open( filename + ".pkl", "wb")
    
    pickle.dump(Object, file)

    file.close()
    

# Open opens filename.pkl from the working directory

def Open(filename):

    file = open(filename + ".pkl", "rb")

    return pickle.load(file)    



# importImage returns an array of pixel values of the dicom image at path

def importImage(path):
    
    image = pydicom.dcmread( path ).pixel_array
    
    return image


# createResDict creates a dictionary of dictionaries. The multi-key is given
# by patient ID and image number pairs. The values are the resolution of the 
# corresponding images.       
    
def createResDict(path):
    
    resDict = {}
    
    for dirName, subDirList, fileList in os.walk(path):
        
        for subDir in subDirList:
            
            folderResDict = {}
            
            
            for file in os.listdir( path + '\\' + subDir ):
                
                image = importImage( createPath(path, subDir, file) )
                
                folderResDict[ remdcm(file) ] = image.shape
                                                                  
                                              
            resDict[subDir] = folderResDict
            
    return resDict
                
# When running this code for the first time, it is necessary to run createResDict.
# It takes a few mins on my computer. To avoid having to do this repeatedly,
# in the code below I save the dictionary outputted by createResDict
#
#
# resDict = createResDict(dataPath)   
# 
# Save(resDict, 'resDict')         
        




# Open resDict.pkl from the working directory

resDict = Open('resDict')




# This function creates a dictionary whose keys are patient IDs. It can be used
# to check if all the images in a given folder are of the same resolution. 
# If they are, then the value of the dictionary is the common resolution of the 
# images in that patient's folder. If they are not, the value is False.


def createResCheck(resDict):
    
    resCheck = {}
    
    
    for Dir in resDict:
        
        Set = set(  resDict[Dir].values()   )
        
        if len(Set) == 1:
            
            resCheck[Dir] = np.squeeze( np.array(list ( Set ) ) )
            
        else:
            
            resCheck[Dir] = False
            
    
    return resCheck
                

resCheck = createResCheck(resDict)


# imageDepth creates a dictionary whose keys are patient IDs. The value is the 
# number of images in the patient's folder, i.e. the 'depth' of the 3D CT scan.

def imageDepth(path):
    
    depthDict = {}
    
    for dirName, subDirList, fileList in os.walk(path):
        
        for subDir in subDirList:
            
            fileCounter = 0
          
            
            for file in os.listdir( path + '\\' + subDir ):
                
                fileCounter += 1    
            
            
            depthDict[subDir] = fileCounter
            
    return depthDict
            
                                                                  
                                              
            
depthDict = imageDepth(dataPath)


# create3dResDict creates a dictionary whose keys are patient IDs.
# Its values are a arrays of the form [imageHeight, imageWidth, imageDepth]

def createResDict3d(resCheck, depthDict):
    

    resDict3d = {}

    
    for Dir in resCheck:
        
        resDict3d[Dir] = np.append(  resCheck[Dir], depthDict[Dir]   )
    

    return resDict3d


resDict3d = createResDict3d(resCheck, depthDict)


# minRes returns the component wise minimum of the 3d resolutions of the images
# in each patient's folder.

def minRes(resDict3d):
    
    resArray = np.empty( [0,3] )
    
    for Dir in resDict3d:
        resArray = np.vstack( (resArray, resDict3d[Dir])  )
    
    
    return np.amin(resArray, axis=0)
        


# resize2dImage alters the resolution of image to desiredRed.
# desiredRes should be a tuple of the form (desiredHeight, desiredWidth)

def resize2dImage(image, desired2dRes):
    
    return resize(image, desired2dRes, anti_aliasing=True)



# approveId outputs a list of patient IDs where the 3d resolution of that 
# patient's images is greater than minres.
# minres should be an np array that is a vector of the form [minHeight, minWidth, minDepth]


def approveID(resDict3d, minres ):
    
    List = []
    
    for ID in resDict3d:
        
        if np.all( resDict3d[ID] >= minres ):
            
            List.append( ID )
        
    return List
 

# res is the 3d resolution that all images will be resized too for use in the NN.
# Change values as desired.

desired3dRes = np.array( [64, 64, 30] )
   


ApprovedIDs = approveID(resDict3d, desired3dRes )


# createImageTensor creates a 3-tensor that stores the 3d image in the directory
# given by ID, with resolution given by desired3dRes. 
# The indices are of the form (height, width, depth)
# That is, first coordinate refers to the height of the pixel, the second to width
# and third to depth.


def createImageTensor(ID, desired3dRes):
    
    desired2dRes = desired3dRes[:2]
    
    desiredDepth = desired3dRes[2]
    
    
    actualDepth = depthDict[ID]
    
   
    imageTensor = np.empty( desired3dRes  )
    
    
    
    
    imageList = list(  resDict[ID].keys() )
    
    imageList.sort()
            
           
    approvedIndices = np.round(   np.linspace(0, (actualDepth-1) , desiredDepth)   ).astype(int)   


    approvedEntries = [  imageList[i] for i in approvedIndices   ]
    
    
    
    for i in range(desiredDepth):
        
        image =  pydicom.dcmread( createPath(dataPath, ID, adddcm( approvedEntries[i] ) ) ).pixel_array
    
        image = resize2dImage(image, desired2dRes)

        imageTensor[:, :, i] = image
        
        
    return imageTensor


# saveImageTensors creates a new folder in D:/Kaggle/code/ named str(desired3dRes)
# Within that folder, it creates 3d image tensors for each patient ID, with 
# resolution desired3dRes.


def saveImageTensors(ApprovedIDs, desired3dRes):
    
    os.chdir('D:\\Kaggle\\code')
    
    dirName = str(desired3dRes)
    
    os.mkdir(dirName  )
    
    os.chdir(dirName)
    
    
    for ID in ApprovedIDs:
        
        Save( createImageTensor(ID, desired3dRes), ID  )
        

# Only need to run saveImageTensors once.        
#  
# saveImageTensors(ApprovedIDs, desired3dRes )   
    



# Read clinical history data into a data frame

clinHist = pd.read_csv("D:\\Kaggle\\train.csv") 



# Add dummy variable that takes value 1 if Sex=Male, 0 otherwise.
# Drop column 'Sex'.

maleDummy = pd.get_dummies( clinHist['Sex'] )['Male']

clinHist = pd.concat( [clinHist, maleDummy] , axis=1  )

clinHist.drop('Sex', axis=1, inplace=True)

# Add two dummy variables that take value 1 if SmokingStatus = 
# 'Ex-smoker', 'Currently smokes', and 0 otherwise.
# Drop column 'SmokingStatus'

smokeDummy = pd.get_dummies( clinHist['SmokingStatus'] )[ ['Ex-smoker','Currently smokes'] ]

clinHist = pd.concat( [clinHist, smokeDummy] , axis=1  )

clinHist.drop('SmokingStatus', axis=1, inplace=True)




# prune removes rows where the patient ID is not in ApprovedIDs

def prune(dataFrame, ApprovedIDs):
    
    return dataFrame[ dataFrame['Patient'].isin(ApprovedIDs)  ]


# prune clinicalHistory     
    
clinHistP = prune(clinHist, ApprovedIDs)    

clinHistP.index = range( clinHistP.shape[0] )




# Given a rowNumber, combineData returns a triple consisting of FVC, the 
# clinical, History data and the image tensor corresponding to that entry.

def combineData( rowNumber ):
    
    y = clinHistP.loc[rowNumber, 'FVC' ]
    
    excludeCols = ['Patient', 'Percent', 'FVC']
    
    includeCols = [col for col in clinHistP.columns if col not in excludeCols ]
    
    xClin = clinHistP.loc[ rowNumber, includeCols ]
    
    ID = clinHistP.loc[rowNumber, 'Patient']
    
    fileName =   'D:\\Kaggle\\code\\'  +  str(desired3dRes) + '\\' + str(ID)
   
    output = [ y,  xClin, Open(fileName) ]
    
    
    return output

# clinDataDim is the dimensionality of the clinical data
# I subtract 2 because one entry is the patient ID, which will not be used,
# and two others are FVC and percent, which are the results of prediction.

clinDataDim = clinHistP.shape[1] - 3

# n is the number of records that will be used in the analysis

n = clinHistP.shape[0]


# This function creates a batch of data to be fed into the NN.
# The batch is a random sample of data records of size batchSize

def createBatch(batchSize):
    
    randomRows = random.sample( range(clinHistP.shape[0]), batchSize)
    
    y = np.empty([batchSize])
    
    xClin = np.empty( [batchSize, clinDataDim] )
    
    xImage = np.empty( [batchSize, 64, 64, 30] )
    
    for i in range(batchSize):
        
        data = combineData( randomRows[i]  )
        
        y[i] = data[0]
        
        xClin[i, :] = data[1]
        
        xImage[i, :, :, :] = data[2]
        
    return (y, xClin, xImage)
    
    
