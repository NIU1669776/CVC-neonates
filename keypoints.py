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

def get_keypoints_static(original_image):
    """
    Detect keypoints in a single, independent image.
    Suitable for folders of images where each image should be processed independently.
    """
    img_mp = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(img_mp)

        key_points = _extract_keypoints(results, img_mp)
        return key_points

# ----------------------------------------
# 2️⃣ Function for sequences (continuous frames)
# ----------------------------------------
def get_keypoints_sequence(original_image, pose=None):
    """
    Detect keypoints in a sequence of images.
    Maintains internal state for smoothing across frames.
    `pose` should be an existing mp_pose.Pose() object with static_image_mode=False.
    """
    if pose is None:
        pose = mp_pose.Pose(static_image_mode=False)

    img_mp = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_mp)

    key_points = _extract_keypoints(results, img_mp)
    return key_points

# ----------------------------------------
# Helper function to extract keypoints
# ----------------------------------------
def _extract_keypoints(results, img_mp):
    """
    Internal helper to extract landmarks into your dict format.
    Returns None if no landmarks detected.
    """
    if results.pose_landmarks:
        height, width, _ = img_mp.shape
        lmk = results.pose_landmarks.landmark

        key_points = {
            "hands": {
                "left_wrist": np.array([lmk[mp_pose.PoseLandmark.LEFT_WRIST].x*width,
                                        lmk[mp_pose.PoseLandmark.LEFT_WRIST].y*height]),
                "right_wrist": np.array([lmk[mp_pose.PoseLandmark.RIGHT_WRIST].x*width,
                                         lmk[mp_pose.PoseLandmark.RIGHT_WRIST].y*height]),
                "left_elbow": np.array([lmk[mp_pose.PoseLandmark.LEFT_ELBOW].x*width,
                                        lmk[mp_pose.PoseLandmark.LEFT_ELBOW].y*height]),
                "right_elbow": np.array([lmk[mp_pose.PoseLandmark.RIGHT_ELBOW].x*width,
                                         lmk[mp_pose.PoseLandmark.RIGHT_ELBOW].y*height])
            },
            "legs": {
                "left_knee": np.array([lmk[mp_pose.PoseLandmark.LEFT_KNEE].x*width,
                                       lmk[mp_pose.PoseLandmark.LEFT_KNEE].y*height]),
                "right_knee": np.array([lmk[mp_pose.PoseLandmark.RIGHT_KNEE].x*width,
                                        lmk[mp_pose.PoseLandmark.RIGHT_KNEE].y*height]),
                "left_ankle": np.array([lmk[mp_pose.PoseLandmark.LEFT_ANKLE].x*width,
                                        lmk[mp_pose.PoseLandmark.LEFT_ANKLE].y*height]),
                "right_ankle": np.array([lmk[mp_pose.PoseLandmark.RIGHT_ANKLE].x*width,
                                         lmk[mp_pose.PoseLandmark.RIGHT_ANKLE].y*height])
            },
            "core": {
                "left_shoulder": np.array([lmk[mp_pose.PoseLandmark.LEFT_SHOULDER].x*width,
                                           lmk[mp_pose.PoseLandmark.LEFT_SHOULDER].y*height]),
                "right_shoulder": np.array([lmk[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*width,
                                            lmk[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*height]),
                "left_hip": np.array([lmk[mp_pose.PoseLandmark.LEFT_HIP].x*width,
                                      lmk[mp_pose.PoseLandmark.LEFT_HIP].y*height]),
                "right_hip": np.array([lmk[mp_pose.PoseLandmark.RIGHT_HIP].x*width,
                                       lmk[mp_pose.PoseLandmark.RIGHT_HIP].y*height]),
                "nose": np.array([lmk[mp_pose.PoseLandmark.NOSE].x*width,
                                  lmk[mp_pose.PoseLandmark.NOSE].y*height])
            }
        }

        gc.collect()
        libc.malloc_trim(0)
        return key_points

    return None