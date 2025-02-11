from facenet_models import FacenetModel
import numpy as np
import config_variables

model = FacenetModel()

def detect_faces(image: np.ndarray):
    boxes, probabilities, landmarks = model.detect(image)
    
    if boxes is not None and len(image.shape) == 3:
        mask = probabilities > config_variables.p_threshold
        return boxes[mask], probabilities[mask], landmarks[mask]
    
    return None, None, None

def get_descriptors(image: np.ndarray, boxes: np.ndarray, probabilities, limit=True):
    descriptors = model.compute_descriptors(image, boxes)

    if limit and descriptors.shape[0] > 1:
        descriptors = np.expand_dims(descriptors[np.argmax(probabilities)], axis=0)
        boxes = np.expand_dims(boxes[np.argmax(probabilities)], axis=0)
        
    return descriptors