import matplotlib.pyplot as plt
import numpy as np
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import matplotlib.pyplot as plt
import cv2
from skimage import io

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    ax.axis('on')
    return fig, ax

def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T

def ellipse_points(resolution, center, x_semiaxis, y_semiaxis):
    """
    Generate points which define a ellipse on an image.

    Centre refers to the centre of the ellipse.
    """   
    radians = np.linspace(0, 2 * np.pi, resolution)
    x = center[0] + x_semiaxis * np.cos(radians)
    y = center[1] + y_semiaxis * np.sin(radians)
    return np.array([x, y]).T

#Example usage

image = io.imread('Filepath/file.png') 
image_gray = color.rgb2gray(image) 

fig, ax = image_show(image); 

elipse = ellipse_points(1000,(1000,1250),800,600)

#ax.plot(elipse[:, 0], elipse[:, 1], '--r', lw=3);

snake = seg.active_contour(image_gray, elipse, alpha=0.095,beta=0.1)

ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);
