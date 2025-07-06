from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
from collections import defaultdict

def preprocess_image(uploaded_file, target_size=(640, 480)):
    image = Image.open(uploaded_file)
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = image.convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    return tf.convert_to_tensor(img_array[tf.newaxis, ...] / 255.0), np.array(image)

def extract_labels(result, threshold=0.3):
    scores = result['detection_scores'].numpy()
    labels = result['detection_class_entities'].numpy()

    label_counts = defaultdict(int)
    for i in range(len(scores)):
        if scores[i] >= threshold:
            label = labels[i].decode('ascii')
            label_counts[label] += 1

    return dict(label_counts)
