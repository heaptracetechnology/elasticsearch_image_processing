# first install numpy -> pip install numpy
# install opencv -> pip install opencv-python

# if any error related to skbuild occues use following commands
# skbuild is for Scikit-build.

# Install it using pip:

# pip install scikit-build

# After the succesfull installation:

# pip install cmake


# if building stuck at installation time
# use this
# pip3 install --upgrade setuptools pip

# pip3 install opencv-python

import numpy as np
import cv2
from numpy.core.fromnumeric import shape

# reading an image
'''
# pixal -> smallest part of image

from numpy.lib.type_check import imag
# reading the image ->
img = cv2.imread('img2.jpeg')  # image read
img2 = cv2.imread(
    r'/home/admin1/Desktop/visibility score/Blur-Detection-Web-App/images/img1.jpeg')  # from other path
cv2.imshow('opimage', img)  # image display
cv2.imshow("secondop", img2)
# kisi key ka interrupt jab tak nahi hota tab tak ye band nahi hota
'''
# wrinting into an image
'''
sample = cv2.imread('sample.jpg')
cv2.imshow('sampleop', sample)  # sample
cv2.imwrite('newsample.jpg', sample)  # clone image into different format
cv2.imwrite('newsample.png', sample)
cv2.waitKey(0)
cv2.destroyAllWindows()  # pure pop ups kill karta hai
'''

'''# getting image info
sample = cv2.imread('sample.jpg')
cv2.imshow('sampleop', sample)
info = sample.shape  # returns tuple
print(info)
print(type(info))
print("height pixel values-->", info[0])
print("width pixel values -->", info[1])
print("image plane-->", info[2])

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# convert colored images into greyscale images
# greyscale images -> blackandwhite images

'''
# 0 paramter pass karne se greyscale me ho jayega
sample = cv2.imread('sample.jpg')
cv2.imshow('sampleop', sample)
grey_img = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
cv2.imshow('grey_img_op', grey_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# convert RGB to Binary Image
'''
# why need to use greyscale image--> colored image is complex to peocess..the rgb colors overlaps over other

sample = cv2.imread('sample.jpg', 0)
cv2.imshow("grey", sample)  # converted to greyscale image
# we use threshhold..if value of pixal is less than threshold then it is black, if above its white
# params-->src, thresh, maxval, type, dst=...)
cv2.waitKey(0)
ret, bw = cv2.threshold(sample, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", bw)
print(ret)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# convert RGB to HSV Color Space


# h -> hue --> point which covers all the color values --> 0 degree/180 degree
# s -> saturation--> covers shade of particular color-->1 -->0-255
# v -> value--> controls intensity of color -->1/2-->0-255

'''
sample = cv2.imread('sample.jpg')
img_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv_image", img_hsv)

cv2.imshow("hue channel", img_hsv[:, :, 0])
cv2.imshow("saturation", img_hsv[:, :, 1])
cv2.imshow("value channel", img_hsv[:, :, 2])

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Extract RGB Color Space

'''sample = cv2.imread('sample.jpg')
cv2.imshow('op', sample)
cv2.waitKey(0)
cv2.destroyAllWindows()

B, G, R = cv2.split(sample)
# apne imahge ka matrix tayar karke..taki masking kr sake
zeros = np.zeros(sample.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("green", cv2.merge([zeros, G, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# x = np.zeros(100)
# print(x.reshape(10, 10))

# Image Translation/image displacement --> left right image rotate karna
'''
sample = cv2.imread('sample.jpg')
print(sample.shape)  # get height and width
height, width = sample.shape[:2]
print(height)
print(width)
quarter_height, quarter_width = height/5, width/5
print(quarter_height)
print(quarter_width)
t = np.float32([[1, 0, quarter_height], [0, 1, quarter_width]])
print(t)

# warpaffine - -> width ans height are dicto exact - -> linear images-->height and width pixals are parallel to rach other
# non-warpaffine - -> thoda tilt ho sakta hai--> non linear images

img_tranlation = cv2.warpAffine(sample, t, (width, height))
cv2.imshow("original image", sample)
cv2.imshow("translation_image->", img_tranlation)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Image Rotation
'''
sample = cv2.imread('sample.jpg')
height, width = sample.shape[:2]
print(height, width)
rotation_matrix = cv2.getRotationMatrix2D(
    (width/4, height/4), 50, .5)  # last one is scale it is used not to lose data while rotation
rotated_image = cv2.warpAffine(sample, rotation_matrix, (width, height))
cv2.imshow('normal_image', sample)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('rotated_image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Image Transpose
'''
# image is treated as a matrix
sample = cv2.imread('sample.jpg')
print(sample.shape)
transposed_image = cv2.transpose(sample)
print(transposed_image.shape)
cv2.imshow('transposed', transposed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# image resizing
'''
sample = cv2.imread('sample.jpg')
# interplotation
cv2.imshow('original image', sample)
# linear interplotation--> size reduce karne ke liye
# cubic interplotation--> size increae karne ke liye
image_reduced = cv2.resize(sample, None, fx=0.25, fy=0.25)
image_increased = cv2.resize(
    sample, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
image_area_wise = cv2.resize(sample, (300, 300), interpolation=cv2.INTER_AREA)
cv2.imshow('scaling--linear interpolation', image_reduced)
cv2.waitKey(0)
cv2.imshow('scaling--cubic interpolation', image_increased)
cv2.waitKey(0)

cv2.imshow("scaling--skewed size", image_area_wise)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''

# Image Pyramid
'''
# without dimension .image ko half ya double kar sakte hai
sample = cv2.imread('sample.jpg')
smaller = cv2.pyrDown(sample)
larger = cv2.pyrUp(sample)
cv2.imshow('normal image', sample)
cv2.waitKey(0)
cv2.imshow('smaller image', smaller)
cv2.waitKey(0)

cv2.imshow('larger image', larger)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''

# Image cropping
'''
sample = cv2.imread('sample.jpg')
height, width = sample.shape[:2]
cv2.imshow('normal image', sample)
cv2.waitKey(0)
# strating parameters of top left
start_row, start_col = int(height*.25), int(width*.25)
# ending parameters of right bottom
end_row, end_col = int(height*.75), int(width*.75)
cropped = sample[start_row:end_row, start_col:end_col]
cv2.imshow('cropped', cropped)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''

# image arithmetics
'''
sample = cv2.imread('sample.jpg')
cv2.imshow('original image', sample)
cv2.waitKey(0)

mat = np.ones(sample.shape, dtype="uint8")*100
add = cv2.add(sample, mat)
cv2.imshow("added", add)
cv2.waitKey(0)
sub = cv2.subtract(sample, mat)
cv2.imshow("substracted", sub)
cv2.waitKey(0)
mul = cv2.multiply(sample, mat)
cv2.imshow("multiplues", mul)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# image bitwise operations
'''
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -1)
cv2.imshow("square", square)
cv2.waitKey(0)

ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow("ellipse", ellipse)

cv2.waitKey(0)

And = cv2.bitwise_and(square, ellipse)
cv2.imshow("and", And)
cv2.waitKey(0)

Or = cv2.bitwise_or(square, ellipse)
cv2.imshow("Or", Or)
cv2.waitKey(0)

xor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("xor", xor)
cv2.waitKey(0)

Not = cv2.bitwise_not(square, ellipse)
cv2.imshow("Not",Not)
cv2.waitKey(0)


cv2.destroyAllWindows()
'''
# image blurring options

'''sample = cv2.imread('sample.jpg')
cv2.imshow('original', sample)
cv2.waitKey(0)

# creating 3*3 kernal
# filter matrix->ye image se pass karvayenge--> to blur image aa jayega
kernel_3 = np.ones((3, 3), np.float32)/9
print(kernel_3)

# kernal sirf odd no. me hi kr sakte hai

kernel_7 = np.ones((7, 7), np.float32)/49
print(kernel_7)
# we use cv2.filter2d to convolve the kernal with an image
# blurred = cv2.filter2D(sample, -1, kernel_3)
blurred = cv2.filter2D(sample, -1, kernel_7)
cv2.imshow("3*3 blurr effect", blurred)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''


# image smoothning

'''
sample = cv2.imread('sample.jpg')
cv2.imshow('original image', sample)
cv2.waitKey(0)

# averaging is done by convolving th eimag ewith a normal box filter,
# this takes thwe pixal under the box and replace the central element
# box size need to odd and positive
blur = cv2.blur(sample, (3, 3))
cv2.imshow('box blur image', blur)
cv2.waitKey(0)

# gaussian blurred -->lighweight hai box se aur blur bhi jayada deta hai and smoothning bhi
gaussian_blur = cv2.GaussianBlur(sample, (7, 7), 0)  # 0 is standar deviation
cv2.imshow('gaussian blur image', gaussian_blur)
cv2.waitKey(0)


# median blur
# take median of all the pixals under the kernal area and central element is replaced with median  values -->
# sare elements ka median calculate karta hai aur jo value aayegi wo center me rakh dega pixal ke
median_blur = cv2.medianBlur(sample, 5)  # 0 is standar deviation
cv2.imshow('median blur image', median_blur)
cv2.waitKey(0)


# bilateral filter --> most effective but heavy graphics card needed

# sigma color and sigma place
bilateral = cv2.bilateralFilter(sample, 9, 75, 75)
cv2.imshow('bilateral blur image', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# image edge detection
'''
sample = cv2.imread('sample.jpg', 0)
height, width = sample.shape[:2]
# 1.sobel technique--> very poor results-->too much noise
sobel_x = cv2.Sobel(sample, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(sample, cv2.CV_64F, 0, 1, ksize=5)
cv2.imshow('normal image', sample)
cv2.waitKey(0)

cv2.imshow('sobel x', sobel_x)

cv2.waitKey(0)
cv2.imshow('sobel y', sobel_y)
cv2.waitKey(0)

sobel_xor = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('sobel_xor image', sobel_xor)
cv2.waitKey(0)

# to enhance the result use laplacian-->but also consist of noise
laplacian = cv2.Laplacian(sample, cv2.CV_64F)
cv2.imshow('laplacian edge detection', laplacian)
cv2.waitKey(0)


# canny edge detection--> best in the class
# uses gradient valures as threshold
canny = cv2.Canny(sample, 20, 150)
cv2.imshow('canny edge detection', canny)
cv2.waitKey(0)
'''
sample = cv2.imread('sample.jpg')
laplacian = cv2.Laplacian(sample, cv2.CV_64F).var()
print(laplacian)
# cv2.imshow('laplacian edge detection', laplacian)
cv2.waitKey(0)
dt = {'laplacian': sample}
cv2.imshow("op image", sample)
print(dt['laplacian'])
cv2.waitKey(0)
cv2.destroyAllWindows()
