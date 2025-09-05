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

libc = ctypes.CDLL("libc.so.6")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def check_borders(coordinate,min_corner,max_corner):
    '''
    Auxiliary function to detect if a given coordinate is inside a bounding box.

    Parameters
    ----------
    coordinate : np.array[2]
        x and y coordinates of the point we intend to decide if its inside our bounding box.

    min_corner : np.array[2]
        Minimum values of the x and y coordinates in the bounding box

    max_corner : np.array[2]
        Maximum values of the x and y coordinates in the bounding box
    '''
    return ((min_corner[0]<coordinate[1]<max_corner[0]) and 
            (min_corner[1]<coordinate[0]<max_corner[1]))

def find_original(img_path):
    '''
    Finds the path corresponding to the non-thermal equivalent given a thermal image path.

    Parameters
    ----------
    img_path : str
        Path of the thermal image

    Returns
    -------
    og_path : str
        Path of the original image

    '''
    img_base_path = img_path[:-5]
    if os.path.isfile(img_base_path+".VIS.jpeg"):
        og_path = img_base_path+".VIS.jpeg"
    elif os.path.isfile(img_base_path+"_VIS.jpeg"):
        og_path = img_base_path+".VIS.jpeg"
    else:
        og_path = img_path
    del img_base_path
    gc.collect()
    libc.malloc_trim(0)
    return og_path

def get_keypoints(original_image):
    '''
    Finds the position of keypoints in a non-thermal image using Mediaipe pose model.

    Parameters
    ----------
    original_image : np.array
        Original non-thermal image of the newborn. We assume it has size 2448x3264 (or 3264x2448)

    Returns
    -------
    key_points : dict
        Dictionary containing the coordinates of each keypoint. In case the model doesnt recognize a pose, it returns None.
        The dict presents the following struct:
            head:{
                nose:[x,y], left_eye:[x,y], right_eye:[x,y], left_ear:[x,y], right_ear:[x,y]
            }
            hands:{
                left_wrist:[x,y], right_wrist:[x,y], left_elbow:[x,y], right_elbow:[x,y]
            }
            legs:{
                left_knee:[x,y], right_knee:[x,y], left_ankle:[x,y], right_ankle:[x,y]
            }
            core:{
                left_shoulder:[x,y], right_shoulder:[x,y], left_hip:[x,y], right_hip:[x,y], chest:[x,y]
            }
    '''
    
    img_mp = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    results = pose.process(img_mp)

    if results.pose_landmarks:

        height,width,_ = img_mp.shape

        # Get landmark coordinates
        landmarks = results.pose_landmarks.landmark

        # Extract specific key points
        key_points = {
            "hands": {
                "left_wrist": np.array((landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x*width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y*height)),
                "right_wrist": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x*width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y*height)),
                "left_elbow": np.array((landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x*width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y*height)),
                "right_elbow": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x*width,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y*height))
            },
            "legs": {
                "left_knee": np.array((landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x*width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y*height)),
                "right_knee": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x*width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y*height)),
                "left_ankle": np.array((landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x*width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y*height)),
                "right_ankle": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x*width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y*height)),
            },
            "core" : {
                "left_shoulder": np.array((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x*width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y*height)),
                "right_shoulder": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*width,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*height)),
                "left_hip": np.array((landmarks[mp_pose.PoseLandmark.LEFT_HIP].x*width,landmarks[mp_pose.PoseLandmark.LEFT_HIP].y*height)),
                "right_hip": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x*width,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y*height)),
                "nose": np.array((landmarks[mp_pose.PoseLandmark.NOSE].x*width,landmarks[mp_pose.PoseLandmark.NOSE].y*height)),
            }    
        }
        
        gc.collect()
        libc.malloc_trim(0)

        return key_points

    del results, img_mp
    gc.collect()
    libc.malloc_trim(0)

    return None