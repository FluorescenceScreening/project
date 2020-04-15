# -*- coding: utf-8 -*-
"""
@author: Marios Theodorou
"""
# =============================================================================
# This is a sample program to measure the intensity of four grey photos from a fluorescent miscroscope analysis. 
# It is based on the Filter program and contains only the image analysis of the origial pictures. 
# Note that Python imports pictures in standard x y 3 format even if they are grey : when grey all three layers are the same (R=G=B)
# The program contains the same functions included in Filter. 
# The pictures are converted to standard RGB, then to CIE 1931 XYZ trichromaticity values, then to xyY where Y is the intensity. 
# The mean of the intensity of each picture and intensity histogram are generated.

# =============================================================================
import colour
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from colour.plotting import * #needed for Chromaticity diagram
# import math
# import scipy.io as sio

# Python requires functions to be defined before calling them. Hence they have to go first in the code
# --------------------------------------------------------------------------------------------------------
def fRGBtoXYZ(RGBimage):
    # Converts image from sRGB to XYZ (CIE 1931 specs)
    # RGBimage has to be a matrix of the shape x y 3, with xy the image pixel addresses for the sRGB values, which must be in the raange 0-1.0
    # http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html
    # This was written based on information from this site to conform to the Matlab transformations. Libraries with standard trasnforms have different approaches.
    RGB_to_XYZ_matrix = np.array([[0.4124564,  0.3575761,  0.1804375],
                                  [0.2126729,  0.7151522,  0.0721750],
                                  [0.0193339,  0.1191920,  0.9503041]])
    v=[0,0,0]
    XYZimage= np.empty_like(RGBimage)
    #xyYimage= np.empty_like(XYZimage)
    dimS=RGBimage.shape # check shape - different from dim

    for i in range(dimS[0]): # loops fom 
        for j in range(dimS[1]):
            for k in range(3):
                V = RGBimage[i,j,k]
                if V<=0.04045:
                    v[k]=V/12.92
                else:
                    v[k]=((V+0.055)/1.055)**2.4
            
            XYZimage[i,j,:]= RGB_to_XYZ_matrix.dot(v)
    return XYZimage

# --------------------------------------------------------------------------------------------------------
def fXYZtoRGB(XYZimage):
    # Converts image from CIE 1931 XYZ to sRGB    
    # XYZimage has to be a matrix of the shape x y 3, with xy the image pixel addresses for the XYZ values, which are in the raange 0-1.0
    # The RGB image values are the rage 0-1.0
    # http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html
    # This was written based on information from this site to conform to the Matlab transformations. Libraries with standard trasnforms have different approaches.
    RGB_to_XYZ_matrix = np.array([[0.4124564,  0.3575761,  0.1804375],
                                  [0.2126729,  0.7151522,  0.0721750],
                                  [0.0193339,  0.1191920,  0.9503041]])
    
    XYZ_to_RGB_matrix=np.linalg.inv(RGB_to_XYZ_matrix) 

    V=[0,0,0]
    RGBimage = np.empty_like(XYZimage)
    dimS=RGBimage.shape # check shape - different from dim
    for i in range(dimS[0]): # loops fom 0-79
        for j in range(dimS[1]):
            V= XYZ_to_RGB_matrix.dot(XYZimage[i,j,:])      
            for k in range(3):
                v = V[k]
                if v<=0.0031308:
                    V[k]=v*12.92
                else:
                    V[k]=1.055*(v)**(1/2.4)-0.055
            RGBimage[i,j,:]=V
    return RGBimage
        
# --------------------------------------------------------------------------------------------------------
def Linearise(imgMatrix):
    # Converts image from shape [x y 3] to 3 linear lists so that they can be plotted. Aligns all the numbers row by row separately for R, G, B
    # The output is a list of 3 linear Numpy arrays  with all the values of R G B 
    sz=imgMatrix.shape
    Linx=np.reshape(imgMatrix[:,:,0],sz[0]*sz[1])
    Liny=np.reshape(imgMatrix[:,:,1],sz[0]*sz[1])
    LinY=np.reshape(imgMatrix[:,:,2],sz[0]*sz[1])   
    LinImg=[Linx,Liny,LinY]
    return LinImg
     
# --------------------------------------------------------------------------------------------------------
def fXYZtoxyY(XYZimage):
    # Converts image from XYZ to xyY
    # Note that the function from colour library, colour.XYZ_to_xyY(imgXYZ) is wrong: it converts [000] to white Y value instead of 0
    xyYimage=colour.XYZ_to_xyY(XYZimage)
    # correct for 000 to get 000 not white
    dim=XYZimage.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if XYZimage[i,j,1]==0:  # If intensity=0 set to 0
                xyYimage[i,j,:]=0
    return xyYimage
# --------------------------------------------------------------------------------------------------------

# MAIN CODE

# Import picture and do all conversions - BGR to RGB, RGB to XYZ, XYZ to xyY, linearize data to use in histogram and plots
g1=cv2.imread('grey1.jpg');
g2=cv2.imread('grey2.jpg');
g4=cv2.imread('grey3.jpg');
g3=cv2.imread('grey4.jpg');

grey1RGB=cv2.cvtColor(g1, cv2.COLOR_BGR2RGB)
grey2RGB=cv2.cvtColor(g2, cv2.COLOR_BGR2RGB)
grey3RGB=cv2.cvtColor(g3, cv2.COLOR_BGR2RGB)
grey4RGB=cv2.cvtColor(g4, cv2.COLOR_BGR2RGB)

grey1XYZ=fRGBtoXYZ(grey1RGB/255)
grey2XYZ=fRGBtoXYZ(grey2RGB/255)
grey3XYZ=fRGBtoXYZ(grey3RGB/255)
grey4XYZ=fRGBtoXYZ(grey4RGB/255)

grey1xyY=fXYZtoxyY(grey1XYZ)
grey2xyY=fXYZtoxyY(grey2XYZ)
grey3xyY=fXYZtoxyY(grey3XYZ)
grey4xyY=fXYZtoxyY(grey4XYZ)

grey1xyYLin=Linearise(grey1xyY)
grey2xyYLin=Linearise(grey2xyY)
grey3xyYLin=Linearise(grey3xyY)
grey4xyYLin=Linearise(grey4xyY)

# Find mean and standard deviation
MeanYgrey1=np.mean(grey1xyY[:,:,2])
MeanYgrey2=np.mean(grey2xyY[:,:,2])
MeanYgrey3=np.mean(grey3xyY[:,:,2])
MeanYgrey4=np.mean(grey4xyY[:,:,2])

# StdDevYgrey1=np.std(grey1xyY[:,:,2])
# StdDevYgrey2=np.std(grey2xyY[:,:,2])
# StdDevYgrey3=np.std(grey3xyY[:,:,2])
# StdDevYgrey4=np.std(grey4xyY[:,:,2])

# PLOTS
# Images - using hstack to stack images horizontally, vstck to stack them vertically. 
dpi1=80
plt.figure(num=0,dpi=dpi1) 
a=np.hstack((grey1RGB,grey2RGB))
b=np.hstack((grey3RGB,grey4RGB))
c=np.vstack((a,b))
plt.axis('off')
plt.title('Pictures [1 2] / [3 4]')
plt.imshow(c)

# Histogram
plt.figure(num=2,dpi=dpi1)  
bin1=50
maxVal=np.round(np.max(grey1xyYLin),1)+0.1
barline=np.linspace(0,maxVal,bin1)
#Plot non zero values 
minY=0.01 
k1=np.where(grey1xyY[:,:,2]>minY)
k2=np.where(grey2xyY[:,:,2]>minY)
k3=np.where(grey3xyY[:,:,2]>minY)
k4=np.where(grey4xyY[:,:,2]>minY)
Y1=grey1xyY[:,:,2]
Y2=grey2xyY[:,:,2]
Y3=grey3xyY[:,:,2]
Y4=grey4xyY[:,:,2]
plt.hist([Y4[k4],Y3[k3],Y2[k2],Y1[k1]], barline, histtype = 'step')
# plt.grid(b=True, which='major', axis='y',color='y')
plt.xlim(0, maxVal)  
plt.title('Intensity values (only >0.01 shown)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend(('Y1', 'Y2', 'Y3','Y4'))   
plt.text( np.amax(Y1)*0.4, Y1.shape[0]*Y1.shape[1]/bin1*5.5, 'Mean Intensity (Y1) = '+ str(round(MeanYgrey1, 5)))
plt.text( np.amax(Y1)*0.4, Y1.shape[0]*Y1.shape[1]/bin1*5.0, 'Mean Intensity (Y2) = '+ str(round(MeanYgrey2, 5)))
plt.text( np.amax(Y1)*0.4, Y1.shape[0]*Y1.shape[1]/bin1*4.5, 'Mean Intensity (Y3) = '+ str(round(MeanYgrey3, 5)))
plt.text( np.amax(Y1)*0.4, Y1.shape[0]*Y1.shape[1]/bin1*4.0, 'Mean Intensity (Y4) = '+ str(round(MeanYgrey4, 5)))
plt.show()