# -*- coding: utf-8 -*-
"""
@author: Marios Theodorou
"""
# This program analyses an imported picture from a fluorescent sample with a background light in a different colour (green and blue used in the demo picture).
# It anlyses the picture by converting the pixel colours to the CIE 1931 colour space.
# In this the RGB corrdinates are transformed to xyY, where xy are the Chromaticity values and Y the intensity.
# This information is then used to structure a filter: The pixel colours are analysed into blue and green intensity fractions and the blue removed, leaving the fluorescent component
# The mean and standard deviation of the intensity of the fluorescense and the relevant chormaticity map and intensity histogram are generated.

import colour
import matplotlib.pyplot as plt
import numpy as np
import cv2
from colour.plotting import * #needed for Chromaticity diagram
import math
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
    xyYimage= np.empty_like(XYZimage)
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
    # Note that the function colour.XYZ_to_xyY(imgXYZ) is wrong: it converts [000] to white 
    xyYimage=colour.XYZ_to_xyY(XYZimage)
    # correct for 000 to get 000 not white
    dim=XYZimage.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if XYZimage[i,j,1]==0:  # If intensity=0 set to black
                xyYimage[i,j,:]=0
    return xyYimage
# --------------------------------------------------------------------------------------------------------
# MAIN CODE

# Import picture and resize if larger size is required. 
imageIn = cv2.imread('pic6.jpg')
# imageIn = cv2.imread('Picture3.png')
# Change from BGR to RGB
imgInRGB = cv2.cvtColor(imageIn, cv2.COLOR_BGR2RGB)
dim0 = (119, 94)# width, height
# resize image
imgInRGB_Resized = cv2.resize(imgInRGB, dim0, interpolation = cv2.INTER_AREA) 
# Convert to CIE1931 xyY
imgXYZ=fRGBtoXYZ(imgInRGB_Resized/255)
imgxyY=fXYZtoxyY(imgXYZ)
imgxyYLin=Linearise(imgxyY)
imgXYZLin=Linearise(imgXYZ)


# CONSTRUCT THE FILTER
# Find gthe location of the max and min intensity point. Max is the green point and min is the blue point
idG=np.amax(np.where(imgxyYLin[2]==np.amax(imgxyYLin[2])))
idB=np.amin(np.where(imgxyYLin[2]==np.amin(imgxyYLin[2])))
# Ideintify the xyY values for the Blue and Green points
greenxyY=[0,0,0]
bluexyY=[0,0,0]
xLin=imgxyYLin[0]
yLin=imgxyYLin[1]
YLin=imgxyYLin[2]
XLin=imgXYZLin[0]
ZLin=imgXYZLin[2]

greenxyY=[xLin[idG], yLin[idG], YLin[idG]]
bluexyY =[xLin[idB], yLin[idB], YLin[idB]]
greenXYZ=[XLin[idG], YLin[idG], ZLin[idG]]
blueXYZ =[XLin[idB], YLin[idB], ZLin[idB]]

# Find weight of Blue for all pixels - 
# Note that for x<Minx the weight is set to 100% This saves a lot of computing power and clears the presentation as most of the pixels are in the left lower corner.
# This min is set at about 25% of the max-min x values. This needs to be reset based on actual experimental results
Factor_settingMinx=0.275
MINx=np.min(imgxyY)+Factor_settingMinx*(np.max(imgxyY)-np.min(imgxyY))
wtBlueLin= np.empty_like(xLin)
for i in range(xLin.size):
    if xLin[i]<MINx:
        wtBlueLin[i]=1.0
    else:
        wtBlueLin[i]=(YLin[i]-greenxyY[2])/(bluexyY[2]-greenxyY[2])
wtBlueRect=np.reshape(wtBlueLin, imgxyY[:,:,0].shape)

# REMOVE THE BLUE. Consider all additional intensity from the green fluorescence 
imgGreenXYZ= np.empty_like(imgXYZ)
imgGreenXYZ[:,:,0]=greenXYZ[0]*(1-wtBlueRect)
imgGreenXYZ[:,:,1]=greenXYZ[1]*(1-wtBlueRect)
imgGreenXYZ[:,:,2]=greenXYZ[2]*(1-wtBlueRect)
# Convert to RGB, xyY and linearise
imgGreenRGB=fXYZtoRGB(imgGreenXYZ)
imgGreenxyY=fXYZtoxyY(imgGreenXYZ)
imgGreenxyYLin=Linearise(imgGreenxyY)

# PLOTS
#  ORIGINAL FIGURE
# Image
dpi1=80
plt.figure(num=0,dpi=dpi1) 
plt.imshow(imgInRGB_Resized)
plt.show()

# Chromaticity diagram
# plt.figure(figsize=[20,12])
plt.figure(num=1, dpi=dpi1) 
plot_chromaticity_diagram_CIE1931(standalone=False)
plt.scatter(imgxyYLin[0],imgxyYLin[1], color='k')
# The next line shows the scatter of x vs Y(intensity) in the same graph if you uncomment
# plt.scatter(imgxyYLin[0],imgxyYLin[2], color='g')
plt.show()

# Histogram 
plt.figure(num=2,dpi=dpi1)  
bin1=50
barline=np.linspace(0,0.8,bin1) 
plt.hist(imgxyYLin, barline, histtype = 'step')
# plt.grid(b=True, which='major', axis='y',color='y')
plt.xlim(0, np.amax(imgxyYLin))  
plt.title('Chromaticity/Intensity xyY values')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend(('Y', 'y', 'x'))   


# FILTER RESULTS

# Weights
# plt.figure(num=3, dpi=dpi1)
# plt.scatter(xLin,wtBlueLin)
# plt.show()

# Image - green fluorescence
plt.figure(num=4, dpi=dpi1)
plt.imshow(imgGreenRGB)  # this shows Green onlyimage
plt.show()

# Chromaticity diagram
plt.figure(num=5, dpi=dpi1) 
plot_chromaticity_diagram_CIE1931(standalone=False)
plt.scatter(imgGreenxyYLin[0],imgGreenxyYLin[1], color='k')
# The next line shows the scatter of x vs Y(intensity) in the same graph if you uncomment
# plt.scatter(imgGreenxyYLin[0],imgGreenxyYLin[2], color='g')
plt.show()

# Histogram 
# Plot only the part which is green, otherwise the histogram is dominated by the black values
NonzeroYflag=np.where(imgGreenxyY[:,:,2])# This returns non zero value locations
NonzeroY=imgGreenxyY[NonzeroYflag]
MeanYGreenNonZero=np.mean(NonzeroY[:,2])
StdDevYGreenNonZero=np.std(NonzeroY[:,2])
MeanYGreen=np.mean(imgGreenxyY[:,:,2])
StdDevYGreen=np.std(imgGreenxyY[:,:,2])

plt.figure(num=6,dpi=dpi1)  
bin1=50
barline=np.linspace(0,0.8,bin1) 
plt.hist(NonzeroY[:,2], barline, rwidth=0.8, histtype = 'bar')
# plt.grid(b=True, which='major', axis='both',color='y')
plt.xlim(0, np.amax(NonzeroY[:,2]))  
plt.title('Intensity Y values')
plt.xlabel('Values')
plt.ylabel('Frequency')
# plt.legend(('Y', 'y', 'x'))   
# The first two values in text are the position, locating it at the mid right side
plt.text( np.amax(NonzeroY[:,2])*0.4, NonzeroY.shape[0]/bin1*3.5, 'Mean Intensity (all pixels) = '+ str(round(MeanYGreen, 4)))
plt.text( np.amax(NonzeroY[:,2])*0.5, NonzeroY.shape[0]/bin1*3.0, 'Standard Deviation = '+ str(round(StdDevYGreen, 3)))
# Note that the mean reported is over all the pixels in the picture. The mean over the non zero ones is very prone to errors depending on the cutoff of the filter

plt.show()
