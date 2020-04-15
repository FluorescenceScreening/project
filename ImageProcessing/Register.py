# -*- coding: utf-8 -*-
"""
@author: Marios Theodorou
"""
# This program imports an image of a well plate and aligns it to a stadard grid. 
# This is the first stage for image analysis, done in other modules with the image exported from this module
# The analysis uses the HoughCircles function to identify circles on the image. This is not able to identify all the circles accurately, so further processing is needed.
# It usually finds about half of those on the standard plate. Using the mean circle diameter of those found, the image is resized to an initial size closer to the grid.
# The standard grid based on geometric measurement of the wells is created as a matrix in Python. The circles found by the HoughCircles are then mapped to the closest ones on the grid.
# A Singular Value Decomposition (SVD) method is then used to determine the angle that the image needs to be rotated.
# This replaced the original bilinear mapping method which determined the rotation angle. SVD is more accurate and much faster.
# The image is then rotated. It is then resized based on the mean distance from the one to the other end of circles - this minimises the error and eliminates the need for loops.
# It is finally translated based on the mean of the error between the circles found and those on the grid.
# The final part creates a binary mask with ones where the circles are. This extracts the circle images. 
# A second mask places bounding boxes around each cirlce and extracts separate pictures for each box into a matrix extractMontage
# extractMontage: structure x,y,3,nWells where xy is the size of the indiviual image for each well, 3 dimensions for RGB and nWells for the number of wells on the plate.


import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import scipy.io as sio

# IMPORT PICTURE then convert to RGB - Python imports to BGR
img = cv2.imread('Wells5r3.JPG')    
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.imread('Well6r.JPG')

# RESCALE TO 1600X image0
scale_0 = 1600/imgRGB.shape[1] # percent of original size
width0 = int(imgRGB.shape[1] * scale_0)
height0 = int(imgRGB.shape[0] * scale_0)
dim0 = (width0, height0) 
 # resize image
image0 = cv2.resize(imgRGB, dim0, interpolation = cv2.INTER_AREA) 

# IDENTIFY CIRCLES - first need to convert to grey
gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
# gray=cv2.medianBlur(gray, 5);
rows = gray0.shape[0]
circles0 = cv2.HoughCircles(gray0, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=20, minRadius=15, maxRadius=40)
# The matrix contains the circle location x, y found and their radii
circles0mean=np.mean(circles0[:,:,2]) 

# RESIZED BASED ON CIRCLE DIAMETER, to get close to standard plate image01
# Resize to 4.5mm diameter = circle_size pixels
circle_radius=36; # suggest use multiple of 9
scale_01=circle_radius/circles0mean;
width01 = int(image0.shape[1] * scale_01)
height01 = int(image0.shape[0] * scale_01)
dim01 = (width01, height01) 
# resize image
image01 = cv2.resize(image0, dim01, interpolation = cv2.INTER_AREA) 

# FIND CIRCLES on image01
gray01 = cv2.cvtColor(image01, cv2.COLOR_RGB2GRAY)
rows = gray01.shape[0]
circles01 = cv2.HoughCircles(gray01, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=20, minRadius=25, maxRadius=50)
circles01mean=np.mean(circles01[:,:,2])
circles01Found=circles01.shape[1] 

# SHOW CIRCLES FOUND ON IMAGE - these are plotted on a TEMP copy of the image
imageTemp= np.copy(image01)

if circles01 is not None:
    circles = np.uint16(np.around(circles01, 0))  
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(imageTemp, center, 1, (0, 100, 100), 5)
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        # circle outline
        radius = i[2]
        cv2.circle(imageTemp, center, radius, (0, 0, 255), 3)

# RESIZE BASED ON VERTICAL DISTANCE BETWEEN FIRST AND LAST ROW image1  
# Identify spacing first
lowerV=0
countL=0
upperV=0
countU=0
for i in range(circles01Found-1):
    lower=np.amin(circles01[:,:,1])//(circle_radius*4)
    upper=np.amax(circles01[:,:,1])//(circle_radius*4)
    if (circles01[:,i,1])//(circle_radius*4)==lower:
        lowerV=lowerV+circles01[:,i,1]
        countL=countL+1
    if (circles01[:,i,1])//(circle_radius*4)==upper:
        upperV=upperV+circles01[:,i,1]
        countU=countU+1
meanSpacing=(upperV/countU-lowerV/countL)/(upper-lower)
# RESCALE
scale_1 =  np.mean(circle_radius*4/meanSpacing)  # percent of original size
width1 = int(image01.shape[1] * scale_1)
height1 = int(image01.shape[0] * scale_1)
dim1 = (width1, height1) 
 # resize image
if scale_1<1.05 and scale_1>0.95:  #Do not rescale if scaling is too extreme 
    image1 = cv2.resize(image01, dim1, interpolation = cv2.INTER_AREA) 

# FIND CIRCLES on image1
gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
rows = gray1.shape[0]
circles1 = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=20, minRadius=25, maxRadius=50)
circles1mean=np.mean(circles1[:,:,2])
circles1Found=circles1.shape[1] 


# PART B: CREATE A STANDARD TEMPLATE AND MATCH CIRCLES TO THAT
# Create a standard template
plateSize=[12, 8]
n1=plateSize[0]
n2=plateSize[1]
nWells=n1*n2
centerwells=np.zeros((2,nWells),dtype=int)
radiiWells = np.zeros((1,nWells),dtype=int)
# Positions it on top left corner based on the most left and top coordinates
centerwells[0,0]=np.around(np.amin(circles1[:,:,0]))
centerwells[1,0]=np.around(np.amin(circles1[:,:,1]))

# Construct circles : spacing between = diameter
k=0
for m in range(n2):
    for n in range(n1):
        centerwells[0,k]= centerwells[0,0]+(n)*circle_radius*4;
        centerwells[1,k]= centerwells[1,0]+(m)*circle_radius*4;
        radiiWells[0,k] = circle_radius
        center=(centerwells[0,k],centerwells[1,k])
        cv2.circle(imageTemp, center, circle_radius, (0, 0, 255), 3)
        k=k+1;

# Apply Singular Value Decomposition to determine Rotation and Translation matrices
# Starting matrix: circles1 without radii, create matrix to nearest Wells :centers_circles1 , matrixNearestWells 
centers_circles1=np.empty((2, circles1Found))
centers_circles1[0,:]=circles1[:,:,0]
centers_circles1[1,:]=circles1[:,:,1]

# match circles found to wells
error=np.ones(circles1Found)*circle_radius*4;
shiftY=np.empty_like(error);
shiftX=np.empty_like(error);
matrixNearestWells=np.empty_like(centers_circles1)

for n in range(circles1Found):    
    for k in range(nWells):   # 1-96
        errorNew=(((centerwells[(0,k)])-centers_circles1[(0,n)])**2+((centerwells[(1,k)])-centers_circles1[(1,n)])**2)**0.5;
        if errorNew<error[n]:
            error[n]=errorNew;
            shiftX[(n)]=centerwells[(0,k)]-centers_circles1[(0,n)];
            shiftY[(n)]=centerwells[(1,k)]-centers_circles1[(1,n)];
            matrixNearestWells[0,n]=centerwells[(0,k)]
            matrixNearestWells[1,n]=centerwells[(1,k)]
er1= centers_circles1 - matrixNearestWells
meanSquareError1=[ (np.mean((er1[0,:])**2))**0.5, (np.mean((er1[1,:])**2))**0.5]  # Euclidian distance to nearest well

cc1=np.copy(centers_circles1)
cc1[0,:]=centers_circles1[0,:]-np.mean(er1[0,:])   #Removes the mean translation error to check improvement with translation only
cc1[1,:]=centers_circles1[1,:]-np.mean(er1[1,:])

er1r= cc1 - matrixNearestWells
meanSquareError1r=[ (np.mean((er1r[0,:])**2))**0.5, (np.mean((er1r[1,:])**2))**0.5]

# CREATE MATRICES FOR SVD - subtract data centroid from circle centres and wells
cc1_c=np.empty_like(cc1)
cc1_c[0,:]=centers_circles1[0,:]-np.mean(centers_circles1[0,:])
cc1_c[1,:]=centers_circles1[1,:]-np.mean(centers_circles1[1,:])

mnw_c=np.empty_like(matrixNearestWells)
mnw_c[0,:]=matrixNearestWells[0,:]-np.mean(matrixNearestWells[0,:])
mnw_c[1,:]=matrixNearestWells[1,:]-np.mean(matrixNearestWells[1,:])

kM=(np.matmul(cc1_c, mnw_c.T))   # CORRELATION MATRIX

kMsvd=np.linalg.svd(kM, full_matrices=True, compute_uv=True, hermitian=False)  # SINGULAR VALUE DECOMPOSITION
# Rotation and translation matrices
rot=np.matmul(kMsvd[0],kMsvd[2])
rotAngle=math.asin(rot[1,0])*360/2/math.pi
trans=[np.mean(matrixNearestWells[0,:]),np.mean(matrixNearestWells[1,:])]-np.matmul(rot, [np.mean(centers_circles1[0,:]),np.mean(centers_circles1[1,:])])

# ROTATE THE IMAGE
rows,cols, col = image1.shape[:]
rotMcv2 = cv2.getRotationMatrix2D((cols/2,rows/2),rotAngle,1)   
# rotation_matrix = cv2.getRotationMatrix2D((image_center_x, image_center_y), -angle, 1)
imgRGBrot = cv2.warpAffine(image1,rotMcv2,(cols,rows))

# RESCALE
scale_2=1.0  #  0.998726  This part is used to check improvements when testing the code. Translation was estimated offline and added here. 
width2 = int(imgRGBrot.shape[1] * scale_2)
height2 = int(imgRGBrot.shape[0] * scale_2)
dim2 = (width2, height2) 
# resize image
imgRGBrotResized = cv2.resize(imgRGBrot, dim2, interpolation = cv2.INTER_AREA) 

# FIND CIRCLES ON NEW IMAGE
gray2 = cv2.cvtColor(imgRGBrotResized, cv2.COLOR_RGB2GRAY)
rows = gray2.shape[0]
circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=20, minRadius=25, maxRadius=50)
circles2mean=np.mean(circles2[:,:,2])
circles2Found=circles2.shape[1] 

# Starting matrix: circles2 without radii, create matrix to nearest Wells :centers_circles1 , matrixNearestWells 
centers_circles2=np.empty((2, circles2Found))
centers_circles2[0,:]=circles2[:,:,0]
centers_circles2[1,:]=circles2[:,:,1]

error2=np.ones(circles2Found)*circle_radius*4;
shiftY2=np.empty_like(error2);
shiftX2=np.empty_like(error2);
matrixNearestWells2=np.empty_like(centers_circles2)

for n in range(circles2Found):
      for k in range(nWells):   # 1-96
        errorNew2=(((centerwells[(0,k)])-centers_circles2[(0,n)])**2+((centerwells[(1,k)])-centers_circles2[(1,n)])**2)**0.5;
        if errorNew2<error2[n]:
            error2[n]=errorNew2;
            shiftX2[(n)]=centerwells[(0,k)]-centers_circles2[(0,n)];
            shiftY2[(n)]=centerwells[(1,k)]-centers_circles2[(1,n)];
            matrixNearestWells2[0,n]=centerwells[(0,k)]
            matrixNearestWells2[1,n]=centerwells[(1,k)]
er2= centers_circles2 - matrixNearestWells2
meanSquareError2=[ (np.mean((er2[0,:])**2))**0.5, (np.mean((er2[1,:])**2))**0.5]

cc2=np.copy(centers_circles2)
cc2[0,:]=centers_circles2[0,:]-np.mean(er2[0,:])
cc2[1,:]=centers_circles2[1,:]-np.mean(er2[1,:])

er2r= cc2 - matrixNearestWells2
meanSquareError2r=[ (np.mean((er2r[0,:])**2))**0.5, (np.mean((er2r[1,:])**2))**0.5]

# TRANSLATE THE IMAGE based on er2
rows1,cols1, col = imgRGBrotResized.shape[:]
# transMcv2=np.empty_like(rotMcv2)
xt=-np.mean(er2[0,:])
yt=-np.mean(er2[1,:])
transMcv2=np.array(([1.0,0.0,xt],[0.0,1.0,yt]))
# img_translation = cv2.warpAffine(image, T, (width, height)) 
imgRGBrotResizedTrans = cv2.warpAffine(imgRGBrotResized,transMcv2,(cols1,rows1))

# SHOW CIRCLES FOUND ON IMAGE
imageTemp2= np.copy(imgRGBrotResizedTrans)

for i in range(nWells):
    center2 = (centerwells[0,i], centerwells[1,i])
    # circle center
    cv2.circle(imageTemp2, center, 1, (0, 100, 100), 5)
    # cv2.circle(image, center_coordinates, radius, color, thickness)
    # circle outline
    radius2 = circle_radius
    cv2.circle(imageTemp2, center2, radius2, (0, 0, 255), 3)

plt.imshow(image01) 
plt.show()
# plt.imshow(gray01)
# plt.show()
plt.imshow(imageTemp)
plt.show()
plt.imshow(imgRGBrotResized)
plt.show()
plt.imshow(imageTemp2)
plt.show()

# EXTRACT THE PICTURES FROM THE IMAGE
# Create a mesh grid with the image size
[ygrid,xgrid]=np.meshgrid(range(rows1), range(cols1),sparse=False, indexing='ij')
# For each circle, create a binary mask with the size of the image with 1s only where the circle is. Use this to extract the circle
mask1=np.empty_like(xgrid)
maskT=xgrid*0
for i in range(nWells):
    mask1 = ((xgrid-centerwells[0,i])**2 + (ygrid-centerwells[1,i])**2) <= circle_radius**2
    maskT = np.add(maskT,mask1)  #  This accumulates all the circles in maskT
    
extractPict1=np.empty_like(imgRGBrotResizedTrans)
extractPict1[:,:,0]=np.multiply(imgRGBrotResizedTrans[:,:,0],maskT)
extractPict1[:,:,1]=np.multiply(imgRGBrotResizedTrans[:,:,1],maskT)
extractPict1[:,:,2]=np.multiply(imgRGBrotResizedTrans[:,:,2],maskT)

# Next, create mask with bounding squares around each circle to extract nWells pictures.
Bbox=np.ones((2,nWells))*circle_radius*2;
# rect_position_and_size=[centerwells-Bbox/2, Bbox]
rect_position=centerwells-Bbox/2
# extractMont is an array [x,y,3, nWells] to be extracted. x y is the size of the picture, 3 for RGB and nWells pictures
extractMont=np.zeros((circle_radius*2,circle_radius*2,3,nWells),dtype=int)
mask2=xgrid*0
for i in range(nWells):
    x1=int(rect_position[0,i])
    y1=int(rect_position[1,i])
    for ydim in range(y1, y1+circle_radius*2):
        for xdim in range(x1, x1+circle_radius*2):
            mask2[ydim,xdim]=1
            extractMont[ydim-y1,xdim-x1,0,i]=extractPict1[ydim,xdim,0]
            extractMont[ydim-y1,xdim-x1,1,i]=extractPict1[ydim,xdim,1]
            extractMont[ydim-y1,xdim-x1,2,i]=extractPict1[ydim,xdim,2]
extractPict2=np.empty_like(imgRGBrotResizedTrans)            
extractPict2[:,:,0]=np.multiply(imgRGBrotResizedTrans[:,:,0],mask2)
extractPict2[:,:,1]=np.multiply(imgRGBrotResizedTrans[:,:,1],mask2)
extractPict2[:,:,2]=np.multiply(imgRGBrotResizedTrans[:,:,2],mask2)



plt.imshow(extractPict1)
plt.show()
plt.imshow(extractPict2)
plt.show()
plt.imshow(extractMont[:,:,:,0])
plt.show()

# save extractMont
# sio.savemat('a.mat', extractMont)
# in matlab: load('a'); c=uint8(a.vect);figure();montage(c); loads the images




