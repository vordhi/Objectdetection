#!/usr/bin/env python
# coding: utf-8



import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util




def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)





labelmap_path = "labelmap.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)





tf.keras.backend.clear_session()
model = tf.saved_model.load(f'inference_graph/saved_model')




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("model_config.config")
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("inference_graph/checkpoint/", 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])



import cv2

cap = cv2.VideoCapture(0)


###########################################################################################
# Uncomment this block of code if you have labview installed and lv_server.vi is running
# import zmq

# context = zmq.Context()

# #  Socket to talk to server
# print("Connecting to Labview…")
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://192.168.122.56:5555") # Labview server running in virtual machine. 
# # If it is running same machine it will be "localhost"
##########################################################################################

import numpy as np

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)

#########################################################################################
 #Uncomment this codeblock if you have labview installed and lv_server.vi is running

    # # Giving alert if there a fire and the threshold is more than 30%
    # arr = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    # score_arr = detections['detection_scores'][0].numpy()

    # if 2 in arr:
    #     index = np.where(arr == 2)
    #     if score_arr[index][score_arr[index]>0.5].any():
    #         socket.send(b"Fire Warning")
    #         message = socket.recv()
    #         print(f"Reply from Labview {message}")
##########################################################################################    
    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






