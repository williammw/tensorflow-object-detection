import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os 

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'Frozen/centernet_hg104_512x512_coco17_tpu-8/saved_model'
print(os.path.isdir(PATH_TO_FROZEN_GRAPH)) # True

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'Labels/label_map.pbtxt'
print(os.path.isfile(PATH_TO_LABELS)) # True
NUM_CLASSES = 90

# Load the frozen detection graph and the label map
loaded_model = tf.saved_model.load(PATH_TO_FROZEN_GRAPH)
infer = loaded_model.signatures['serving_default']

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Open a video file or a camera stream
cap = cv2.VideoCapture(0) # Use 0 for built-in camera or change to a video file path

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Run detection
    detections = infer(tf.convert_to_tensor(image_np_expanded))

    # Visualize the results of the detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.60)

    # Display the resulting image
    cv2.imshow('object detection', cv2.resize(image_np, (480, 320)))

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
