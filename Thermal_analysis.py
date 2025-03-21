import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import easyocr
import mediapipe as mp
import gc
import cupy as cp
import glob
import json
import ctypes

libc = ctypes.CDLL("libc.so.6")

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

    return index_matrix

# He tenido que usar la versión 11.0.0 de pillow porque no funciona sino
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
    reader = easyocr.Reader(['es'])
    
    if image.shape == (480,640,3):
        box_1 = image[141:168,25:75]
        box_2 = image[393:420,25:75]
    else:
        box_1 = image[141:168,25:75]
        box_2 = image[458:485,25:75]

    top = float(reader.readtext(box_1,allowlist='0123456789.',detail=0)[0])
    bot = float(reader.readtext(box_2,allowlist='0123456789.',detail=0)[0])
    return top,bot

def temp_classifier(image):
    """
    Extracts the temperature for each pixel of the image and converts it locally to grayscale.

    Parameters
    ----------
    image : np.array
        The image to transform. We assume is taken by a thermal camera with resolution 480*640 with gbr values.
    
    Returns
    --------
    index_matrix : np.array
        Grayscale converted image using the lateral thermal bar as reference.
    
    temps : np.array
        Matrix containing the temperature of each pixel based on the temperature benchmark shown on the image.
    """
    bar = get_bar(image)
    np_img = np.array(image).reshape(image.shape[0]*image.shape[1],3).astype(float)
    distances = np.linalg.norm(np_img[:,np.newaxis,:]-bar[np.newaxis,:,:],axis=2)
    
    indices = np.ones(np_img.shape[:-1])*255 - np.argmin(distances,axis=1)*255/bar.shape[0]
    index_matrix = indices.reshape(image.shape[0],image.shape[1])

    # Extraer temperatura
    TOP, BOT = limit_finder(image)
    temps = index_matrix*(TOP-BOT)/255 + BOT
    return index_matrix, temps

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

    libc.malloc_trim(0)

    return index_matrix.get(), temps.get()

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
    del og_path
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

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    results = pose.process(img_mp)

    if results.pose_landmarks:

        height,width,_ = img_mp.shape
        # Get landmark coordinates
        landmarks = results.pose_landmarks.landmark
            
        # Extract specific key points
        key_points = {
            "head": {
                "nose": np.array((landmarks[mp_pose.PoseLandmark.NOSE].x*width, landmarks[mp_pose.PoseLandmark.NOSE].y*height)),
                "left_eye": np.array((landmarks[mp_pose.PoseLandmark.LEFT_EYE].x*width, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y*height)),
                "right_eye": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x*width, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y*height)),
                "left_ear": np.array((landmarks[mp_pose.PoseLandmark.LEFT_EAR].x*width, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y*height)),
                "right_ear": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x*width, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y*height)),
            },
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
                "right_hip": np.array((landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x*width,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y*height))
            }    
        }
        key_points["core"]["chest"] = (key_points["core"]["left_hip"]*0.15+key_points["core"]["right_hip"]*0.15 + key_points["core"]["left_shoulder"]*0.35 + key_points["core"]["right_shoulder"]*0.35)
        return key_points
    return None
    
def get_keypoint_temperature (img_thermal, img_og):
    '''
    Extracts the temperature of the keypoints of a thermal image. 
    
    For this to work properly. Both photos must be taken from the same position and in the same moment. The 
    difference in the FOV (Field Of View) between both thermal and non thermal images is considered to be a 
    fixed value in line 296, in case to be different, the adjustments to the keypoints must be edited

    Parameters
    ----------
    img_thermal : np.array
        Thermal image of the newborn. We assume it is taken by a thermal camera with resolution 480x640 (640x480) 
        with gbr values.
    img_og : np.array
        Orignial image of the newborn. We assume it is taken by a normal camera with resolution 2448x3264 (or 3264x2448) 
        with gbr values.

    Returns
    -------
    temp_vals : dict
        Dictionary containing each keypoint as key and the respective temperature as value. If there is no baby, it 
        returns None. If a point is out of frame, that value of the dictionaty will be None.
    derived_kpoints : dict
        Dictionary containing the coordinates for each keypoint in the thermal image. If there is no baby, it returns
        None. Same structure as key_points from function get_keypoints().
    '''
    keypoints = get_keypoints(img_og)
    derived_kpoints = keypoints
    if derived_kpoints != None:
        for sec_name, section in keypoints.items():
            for name,coord in section.items(): # Crop applied: [345:2055,500:2900,:], New dimensions: (480,640)
                new_x = (coord[0]-500)*0.2667
                new_y = (coord[1]-245)*0.2657
                derived_kpoints[sec_name][name] = np.array([new_x,new_y])
    else:
        print("No hay bebé")
        return None, None

    _, temps = temp_classifier_gpu(img_thermal)
    temp_vals = {}
    for section in derived_kpoints.values():
        for name,coord in section.items():
            if check_borders(coord,(0,0),(img_thermal.shape[0],img_thermal.shape[1])):
                temp_vals[name] = round(temps[int(coord[1])][int(coord[0])],3)
                if temp_vals[name]>=max(30,np.max(temps)-5):
                    print("Temperatura",name,":",temp_vals[name])
                else:
                    print("Temperatura",name,":",temp_vals[name],"(No concluyente)")
                    #temp_vals[name] = None
            else:
                print("Temperatura",name,": Se sale de la imagen")
                temp_vals[name] = None

    del keypoints
    libc.malloc_trim(0)
    return temp_vals, derived_kpoints


# Define the folder paths
folders = [
    "images/40/24.10.24-25.10.24/",
    "images/40/25.10.24/",
    "images/40/25.10.24 (2)/"
]
print("Check 1")
# List to store valid image paths
valid_images = []

# Iterate through the folders
for folder in folders:
    # Get all .jpeg files in the folder
    for file in glob.glob(os.path.join(folder, "*.jpeg")):
        # Exclude files ending in .VIS.jpeg or _VIS.jpeg
        if not (file.endswith(".VIS.jpeg") or file.endswith("_VIS.jpeg")):
            valid_images.append(file)


# Print the valid image file names
for path in valid_images:
    vals = {}
    og_path = find_original(path)
    img = cv2.imread(path)
    og_im = cv2.imread(og_path)
    t, pos = get_keypoint_temperature(img,og_im)
    if t != None:
        for key in t.keys():
            if key in vals.keys():
                vals[key].append(t[key])
            else:
                vals[key] = [t[key]]
    with open("data1.json","a") as f:
        json.dump(vals,f)
    del vals,og_path,img,t,pos,og_im
    libc.malloc_trim(0)