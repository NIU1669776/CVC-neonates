import numpy as np
import cv2
import matplotlib.pyplot as plt
from thermal import temp_classifier_gpu as temp_classifier
from keypoints import get_keypoints_sequence, get_keypoints_static, check_borders
import os

def rgb_thermal_projection(x_big, y_big):
    """
    Maps a point (x_big, y_big) from the big RGB image to the small (cropped & resized) image.
    Returns (x_small, y_small) or None if point is outside the crop.


    Crop and small image sizes are hardcoded from previous calculations.
    """
    # --- Crop coordinates ---
    x0, y0 = 500, 350
    x1, y1 = 2745, 2034 # x1 = x0 + 2245, y1 = y0 + 1684


    # --- Small image size ---
    W_small, H_small = 640, 480


    # Check if point is inside the crop
    if x_big < x0 or x_big > x1 or y_big < y0 or y_big > y1:
        return None, None # Point is outside the crop


    # Shift to crop coordinates
    x_crop = x_big - x0
    y_crop = y_big - y0


    # Scale to small image
    scale_x = W_small / (x1 - x0)
    scale_y = H_small / (y1 - y0)


    x_small = x_crop * scale_x
    y_small = y_crop * scale_y


    return x_small, y_small

def get_keypoint_temperature (img_thermal, img_og, sequence_pose=None):
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

    # intentar detectar keypoints directamente
    if sequence_pose is None:
        raw_keypoints = get_keypoints_static(img_og)
    else:
        raw_keypoints = get_keypoints_sequence(img_og,sequence_pose)

    print("DEBUG - raw keypoints from get_keypoints:", raw_keypoints)

    keypoints = raw_keypoints
    if keypoints is None:
        print("No keypoints found in the original image.")
        return None, None
    
    else:

        colors = {
            "hands": 'red',
            "legs": 'green',
            "core": 'blue',
            "head": 'yellow'
        }

        #print("Keypoints found in the original image:", keypoints)

        # Map keypoints to thermal image coordinates

        derived_kpoints = keypoints.copy()
        if derived_kpoints != None:
            for sec_name, section in keypoints.items():
                for name,coord in section.items(): # 
                    new_x, new_y = rgb_thermal_projection(coord[0],coord[1])
                    derived_kpoints[sec_name][name] = np.array([new_x,new_y])

            _, temps = temp_classifier(img_thermal)


            if temps is not None:
                temp_vals = {}
                for section in derived_kpoints.values():
                    for name,coord in section.items():
                        if coord[0] is not None and coord[1] is not None:
                            print("DEBUG - Mapped coord for", name, ":", coord)
                            print("DEBUG - Image shape:", img_thermal.shape)
                            print("DEBUG - Temp map", temps)
                            temp_vals[name] = round(temps[int(coord[1])][int(coord[0])],3)

                            '''
                            if temp_vals[name]>=max(30,np.max(temps)-5):
                                #print("Temperatura",name,":",temp_vals[name])
                            else:
                                print("Temperatura",name,":",temp_vals[name],"(No concluyente)")
                                #temp_vals[name] = None
                            '''
                        else:
                            #print("Temperatura",name,": Se sale de la imagen")
                            temp_vals[name] = None
            else:
                temp_vals = None
        else:
            temp_vals, derived_kpoints = None, None

        del keypoints, img_thermal, img_og
        return temp_vals, derived_kpoints
    
if __name__ == "__main__":
    thermal_img_path = "images/40/25.10.24 (2)/HM20241025150105.jpeg"
    og_img_path = "images/40/25.10.24 (2)/HM20241025150105.VIS.jpeg"

    thermal_img = cv2.imread(thermal_img_path)
    og_img = cv2.imread(og_img_path)
    if not os.path.exists(thermal_img_path):
        print(f"Thermal image not found at {thermal_img_path}")
        exit()

    
    plt.imshow(cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB))
    plt.show()

    if not os.path.exists(og_img_path):
        print(f"Original image not found at {og_img_path}")
        exit()

    temps, kpoints = get_keypoint_temperature(thermal_img, og_img)
    print("DEBUG - Temperatures at keypoints:", temps)
    print("DEBUG - Keypoints in thermal image:", kpoints)

    # Plot the thermal image with the derived keypoints
    show = "Thermal"  # Change to "Original" to see the original image with keypoints
    if show == "Thermal":
        if kpoints != None:
            plt.imshow(cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB))
            colors = {
                "hands": 'red',
                "legs": 'green',
                "core": 'blue',
                "head": 'yellow'
            }
            for section_name, section_coords in kpoints.items():
                for key, coord in section_coords.items():
                    if isinstance(coord, np.ndarray):
                        plt.scatter(coord[0], coord[1], label=f"{section_name}-{key}", s=50, c=colors[section_name], marker='x')
            plt.title("Thermal Image with Derived Keypoints")
            plt.show()
    elif show == "Original":
        kpoints = get_keypoints_static(og_img)
        if kpoints != None:
            plt.imshow(cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB))
            colors = {
                "hands": 'red',
                "legs": 'green',
                "core": 'blue',
                "head": 'yellow'
            }
            for section_name, section_coords in kpoints.items():
                for key, coord in section_coords.items():
                    if isinstance(coord, np.ndarray):
                        plt.scatter(coord[0], coord[1], label=f"{section_name}-{key}", s=50, c=colors[section_name], marker='x')
            plt.title("Original Image with Derived Keypoints")
            plt.show()
    del thermal_img, og_img