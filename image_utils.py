from imageio.v2 import imread
from model_utils import detect_faces, get_descriptors
import numpy as np
import os
import config_variables
from PIL import Image


def cos_dist(d1, d2):
    return 1- np.matmul(d1, d2.T) / (np.linalg.norm(d1, axis=1, keepdims=True)*np.linalg.norm(d2, axis=1, keepdims=True).T)

def read_image(path: str): 
    """
    Returns a NumPy array of the RGB data 
    of the image stored at path.
    
    Parameters
    ----------
    path : str
        A path-like string where the image is stored

    Returns
    -------
    np.ndarray - shape(image_height, image_width, 3)
        Array of RGB values for the image
    """
    image = imread(str(path))
    if image.shape[-1] == 4:
        image = image[..., :-1]
    return image

def censor_image(image, database):
    censored = image.copy()
    
    boxes, probabilities, landmarks = detect_faces(image)

    if boxes is not None:
        descriptors = get_descriptors(image, boxes, probabilities, limit=False)
        dists = cos_dist(descriptors, np.array(database))
        
        for box in boxes[np.unique(np.argwhere(dists < config_variables.dist_threshold)[:, 0])]:
            x0, y0, x1, y1 = box
            censored[int(y0):int(y1), int(x0):int(x1)] = [0, 0, 0]
            
    return censored

def censor_directory(image_dir_path: str, save_dir_path: str, database):
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        
    for file in os.listdir(image_dir_path):
        censored_image = censor_image(read_image(os.path.join(image_dir_path, file)), database)
        Image.fromarray(censored_image).save(os.path.join(save_dir_path, file))
    