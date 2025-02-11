from bing_image_downloader import downloader
from image_utils import read_image
from model_utils import detect_faces, get_descriptors
import numpy as np
import pickle
import os

def save_database(file_path: str, database: dict):
    with open(file_path, "wb") as file:
        pickle.dump(database, file)

def load_database(file_path: str):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def populate_database(names: list, limit: int, image_dir: str):
    database = []

    #clear images directory
    for f in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, f))

    for name in names:
        downloader.download(name, limit=limit, output_dir=image_dir, verbose=False)
            
        # Read images from directory
        images = [read_image(os.path.join(os.path.join(image_dir, name), file)) for file in os.listdir(os.path.join(image_dir, name))]

        # Calculate descriptors 
        image_descriptors = []
        for image in images:
            if len(image.shape) == 3:
                boxes, probabilities, landmarks = detect_faces(image)
            else:
                boxes = None

            if boxes is not None and len(boxes) != 0:
                descriptors = get_descriptors(image, boxes, probabilities)

                for descriptor in descriptors:
                    image_descriptors.append(descriptor)

        average_descriptor = np.mean(image_descriptors, axis=0)
        database.append(average_descriptor)
        
    return database

def add_to_database(database: list, name: list, limit: int, image_dir: str):
    #clear images directory
    #for f in os.listdir(image_dir):
    #    os.remove(os.path.join(image_dir, f))

    downloader.download(name, limit=limit, output_dir=image_dir, verbose=False)
            
    # Read images from directory
    images = [read_image(os.path.join(os.path.join(image_dir, name), file)) for file in os.listdir(os.path.join(image_dir, name))]

    # Calculate descriptors 
    image_descriptors = []
    for image in images:
        if len(image.shape) == 3:
            boxes, probabilities, landmarks = detect_faces(image)
        else:
            boxes = None

        if boxes is not None:
            descriptors = get_descriptors(image, boxes, probabilities)

            for descriptor in descriptors:
                image_descriptors.append(descriptor)

    average_descriptor = np.mean(image_descriptors, axis=0)
    database.append(average_descriptor)