import os
import numpy as np
import tensorflow as tf
from cv2 import cv2
from keras.models import load_model


def load_and_preprocess_image(image, mean_gray=4.864612497788194934e-01):
    # load image from file path
    # image = tf.io.read_file(image_path)
    # decode jpeg encoded image
    image = tf.image.decode_jpeg(image, channels=1)
    # normalize pixel values to be in the range [0, 1] and subtract mean intensity
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, mean_gray)
    return image


# loading model
model = None

if os.path.isfile('./models/model_checkpoint.h5'):
    print("found checkpoint, loading model")
    model = load_model('./models/model_checkpoint.h5')
else:
    print("Model Checkpoint not found. Exit")

downSample = 4
color = (0, 0, 255)
thickness = 3
radius = 30

# setup video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hg, wd = gray.shape
    gray = cv2.resize(gray, (int(wd / downSample), int(hg / downSample)))
    # gray_tf = load_and_preprocess_image(gray)
    t_gray = cv2.transpose(gray)
    g_tensor = tf.convert_to_tensor(t_gray, dtype=tf.float32)
    g_tensor = tf.expand_dims(g_tensor, 0)

    x = model.predict(g_tensor)
    print(x)
    # Display the resulting frame
    coord = (int(x[0][0][0]), int(x[1][0][0]))
    gray = cv2.circle(gray, coord, radius, color, thickness)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
c

