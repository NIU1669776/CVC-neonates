import cv2
import numpy as np
import os, gc
import easyocr
import mediapipe as mp
import cupy as cp
import glob
import json
import ctypes, tracemalloc
import psutil
import matplotlib.pyplot as plt


cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

libc = ctypes.CDLL("libc.so.6")
reader = easyocr.Reader(['es'])

def get_bar(img):
    '''
    Extracts the bar in the image in order to find the thermal scale

    Parameters
    ----------
    img : np.array
        The image to extract the bar from. We assume is taken by a 480x640 (or 640x480).

    Returns
    -------
    bar : np.array
        The bar cropped from the original image.
    '''
    if img.shape == (480,640,3):
        bar = img[182:381,16]
    else:
        bar = img[181:446,16]
    return bar

def opt2(image):
    '''
    Transforms the image into an apropiate gray scale considering the thermal gradient shown on the lateral bar.T

    Parameters
    ----------
    image : np.array
        The image to transform. We assume is taken by a thermal camera with resolution 480x640 with rgb values.

    Returns
    --------
    index_matrix : np.array
        Grayscale converted image using the lateral thermal bar as reference.
    '''
    bar = get_bar(image)
    np_img = np.array(image).reshape(image.shape[0]*image.shape[1],3).astype(float)
    distances = np.linalg.norm(np_img[:,np.newaxis,:]-bar[np.newaxis,:,:],axis=2)
    
    indices = np.ones(np_img.shape[:-1])*255 - np.argmin(distances,axis=1)*255/bar.shape[0]
    index_matrix = indices.reshape(image.shape[0],image.shape[1])
    del bar, np_img, distances, indices
    libc.malloc_trim(0)
    return index_matrix

def limit_finder(image):
    """
    This function extracts the information shown of the boxes of the photo. The top one shows the maximum temperature and the bottom one the minimum.

    Parameters
    ----------
    image : np.array
        The image to transform. We assume is taken by a thermal camera with resolution 480*640 with rgb values.

    Returns:
    ---------
    top : float
        Highest temperature in the image. Corresponds to the index 0 of the thermal gradient.

    bot : float
        Lowest temperature in the image. Corresponds to the index -1 of the thermal gradient.
    """
    if image.shape == (480,640,3):
        box_1 = image[141:168,25:75]
        box_2 = image[393:420,25:75]
    else:
        box_1 = image[141:168,25:75]
        box_2 = image[458:485,25:75]

    top = float(reader.readtext(box_1,allowlist='0123456789.',detail=0)[0])
    bot = float(reader.readtext(box_2,allowlist='0123456789.',detail=0)[0])
    del box_1, box_2
    return top,bot

def temp_classifier_gpu(image):
    """
    Extracts the temperature for each pixel of the image and converts it locally to grayscale.

    Parameters
    ----------
    image : np.array
        The image to transform. We assume is taken by a thermal camera with resolution 480*640 with rgb values.

    Returns
    --------
    index_matrix : np.array
        Grayscale converted image using the lateral thermal bar as reference.
    
    temps : np.array
        Matrix containing the temperature of each pixel based on the temperature benchmark shown on the image.
    """
 
    bar = cp.asarray(get_bar(image))
    cp_img = cp.asarray(image).reshape(image.shape[0]*image.shape[1],3).astype(float)
    distances = cp.linalg.norm(cp_img[:,cp.newaxis,:]-bar[cp.newaxis,:,:],axis=2)
    
    indices = cp.ones(cp_img.shape[:-1])*255 - cp.argmin(distances,axis=1)*255/bar.shape[0]
    index_matrix = indices.reshape(image.shape[0],image.shape[1])

    # Extraer temperatura
    TOP, BOT = limit_finder(image)
    if TOP>=100: TOP=TOP/10
    if BOT>=100: BOT=BOT/10
    temps = index_matrix*(TOP-BOT)/255 + BOT

    del cp_img, distances, indices, bar #Lo he añadido a ver si se arregla lo de la memoria

    cp.get_default_memory_pool().free_all_blocks()
    return index_matrix.get(), temps.get()

if __name__ == "__main__":
    print("Version de Python: ", os.sys.version)
    print("Version de Cupy: ", cp.__version__)
    print("Version de OpenCV: ", cv2.__version__)
    print("Version de Numpy: ", np.__version__)
    print("Version de EasyOCR: ", easyocr.__version__)
    print("Version de Mediapipe: ", mp.__version__)
    print("Version de OS: ", os.name)
    print("Get Cuda Device Count:",cp.cuda.runtime.getDeviceCount())