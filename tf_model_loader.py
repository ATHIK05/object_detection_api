import tensorflow as tf
import tensorflow_hub as hub

print("Loading model...")
detector = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures['default']
print("Model loaded.")

def load_model():
    return detector
