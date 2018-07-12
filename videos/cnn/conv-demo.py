from PIL import Image
import numpy
from scipy.signal import convolve2d
import cv2

kernel = [[0.1,0.1,0.1],
          [0.1,0.1,0.1],
          [0.1,0.1,0.1]]

backgroundColor = (0,)*3
pixelSize = 10
imgFile = 'puppy.jpg'
color = True

def drawImage(channel):
  image = Image.open(imgFile)
  red, green, blue = image.split()
  if channel == 'grey':
    image = Image.open(imgFile).convert('LA')
  elif channel == 'r':
    image = red
  elif channel == 'g':
    image = green
  elif channel == 'b':
    image = blue


  image = image.resize((image.size[0]//pixelSize, image.size[1]//pixelSize), Image.NEAREST)
  image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
  image = image.convert('RGB')
  pixel = image.load()

  
  for i in range(0,image.size[0],pixelSize):
    for j in range(0,image.size[1],pixelSize):
      for r in range(pixelSize):
        pixel[i+r,j] = backgroundColor
        pixel[i,j+r] = backgroundColor

        
  return image

def drawImageConv(kernel, x, y):
  image = Image.open(imgFile)
  image = image.resize((image.size[0]//pixelSize, image.size[1]//pixelSize), Image.NEAREST)

  new_image = convolve2d(numpy.asarray(image)[:,:,0], kernel)
  new_image = new_image.clip(0.0, 255.0)
  for i in range(new_image.shape[0]):
    for j in range(new_image.shape[1]):
      if (i>y or (i==y and j>=x)):
        new_image[i,j] = 0

  
  image = Image.fromarray(new_image)
  image = image.convert('RGB')
  image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
  pixel = image.load()

  
  for i in range(0,image.size[0],pixelSize):
    for j in range(0,image.size[1],pixelSize):
      for r in range(pixelSize):
        pixel[i+r,j] = backgroundColor
        pixel[i,j+r] = backgroundColor

    
  return image

def showImage(x,y):
  if color:
    imageR= drawImage('r')
    imageG= drawImage('g')
    imageB= drawImage('b')
    opencvImageR = cv2.cvtColor(numpy.array(imageR), cv2.COLOR_RGB2BGR)
    opencvImageG = cv2.cvtColor(numpy.array(imageG), cv2.COLOR_RGB2BGR)
    opencvImageB = cv2.cvtColor(numpy.array(imageB), cv2.COLOR_RGB2BGR)
    
    convImage = drawImageConv(kernel, x, y)
    opencvConvImage = cv2.cvtColor(numpy.array(convImage), cv2.COLOR_RGB2BGR)

    cv2.rectangle(opencvImageR, (x*pixelSize,y*pixelSize), ((x+w)*pixelSize,(y+h)*pixelSize), (0,0,255))
    cv2.rectangle(opencvImageG, (x*pixelSize,y*pixelSize), ((x+w)*pixelSize,(y+h)*pixelSize), (0,0,255))
    cv2.rectangle(opencvImageB, (x*pixelSize,y*pixelSize), ((x+w)*pixelSize,(y+h)*pixelSize), (0,0,255))
    cv2.imshow('image red', opencvImageR)
    cv2.imshow('image green', opencvImageG)
    cv2.imshow('image blue', opencvImageB)
    cv2.imshow('image out', opencvConvImage)
  else:
    image= drawImage('grey')
    opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    
    convImage = drawImageConv(kernel, x, y)
    opencvConvImage = cv2.cvtColor(numpy.array(convImage), cv2.COLOR_RGB2BGR)

    cv2.rectangle(opencvImage, (x*pixelSize,y*pixelSize), ((x+w)*pixelSize,(y+h)*pixelSize), (0,0,255))
    cv2.imshow('image', opencvImage)
    cv2.imshow('image conv', opencvConvImage)




x = 0
w = 3
h = 3
y = 0




while(True):

  k = cv2.waitKey(0)
  if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
    break
  elif k == 32:
    showImage(100, 100)
  else:
    showImage(x,y)
    x+=1
    if (x > 32):
      x = 0
      y += 1
    


