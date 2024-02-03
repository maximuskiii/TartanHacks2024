# import packages
import math

from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
import scipy.sparse as sp
from vidgear.gears import CamGear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.6, help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Generate random COLORS for each class label
COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Lime
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (192, 192, 192), # Silver
    (128, 0, 0),     # Maroon
    (128, 128, 0),   # Olive
    (0, 128, 0),     # Green
    (128, 0, 128),   # Purple
    (105, 105, 105), # Dim Gray
    (0, 0, 128),     # Navy
    (255, 165, 0),   # Orange
    (255, 215, 0),   # Gold
    (66, 206, 245),   # Teal
    (0, 100, 0),     # Dark Green
    (139, 0, 139),   # Dark Magenta
    (85, 107, 47),   # Dark Olive Green
    (255, 69, 0),    # Orange Red
    (75, 0, 130)     # Indigo
]

def sinusoidal_scoring(num_people):
    if num_people <= 0:
        return 0

    # Constants to control the behavior of the function
    amplitude = 5  # Controls the peak value of the sine wave
    frequency = 0.5  # Controls the frequency of the sine wave
    log_base = 10  # Controls the rate of logarithmic growth

    # Apply a logarithmic transformation followed by a sine transformation
    score = amplitude * math.sin(frequency * math.log(num_people, log_base))
    return score

def create_zero_matrix(n, m):
    return np.zeros((n, m))

def fill_matrix_with_value(matrix, start_x, start_y, end_x, end_y, value):
    n = matrix.shape[0]
    # Fill the specified area with ones
    matrix[start_x:end_x, start_y:end_y] = value
    return matrix

# Function to process each section of the frame
def process_frame_section(prototxt, model, frame_section, args):
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    (h, w) = frame_section.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_section, 1 / 127.5, (w, h), 127.5)
    net.setInput(blob)
    predictions = net.forward()

    detections = []
    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        class_id = int(predictions[0, 0, i, 1])
        if confidence > args["confidence"]:
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detections.append((startX, startY, endX, endY, confidence, class_id))
    return detections

# Function to combine results from all sections
def combine_results(sections_results, num_rows, num_cols, box_height, box_width):
    combined_results = []
    for row in range(num_rows):
        for col in range(num_cols):
            start_y = row * box_height
            start_x = col * box_width

            # Adjusting and adding detections from each box
            for detection in sections_results[row * num_cols + col]:
                x_min, y_min, x_max, y_max, confidence, class_id = detection

                # Adjust coordinates to map to the position in the original frame
                x_min += start_x
                x_max += start_x
                y_min += start_y
                y_max += start_y

                combined_results.append((x_min, y_min, x_max, y_max, confidence, class_id))

    return combined_results

# Load the model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("/Users/bkleyner/Desktop/dronefootage.mp4")
#vs = CamGear(source='https://www.youtube.com/watch?v=WvhYuDvH17I&ab_channel=masterryze', stream_mode=True).start()
# use 0 for webcam, or replace with video file path
fps = FPS().start()

# Process each frame
while True:
    ret, frame = vs.read()
    if not ret:
        break
    #frame = vs.read()

    frame = imutils.resize(frame, width=2160)
    (h, w) = frame.shape[:2]
    num_rows, num_cols = 2, 3
    box_height, box_width = h // num_rows, w // num_cols

    # Process each frame section in parallel
    with ThreadPoolExecutor(max_workers=num_rows * num_cols) as executor:
        futures = []
        for row in range(num_rows):
            for col in range(num_cols):
                start_y = row * box_height
                end_y = (row + 1) * box_height
                start_x = col * box_width
                end_x = (col + 1) * box_width
                frame_box = frame[start_y:end_y, start_x:end_x]
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # Green color, 2px thickness
                futures.append(executor.submit(process_frame_section, args["prototxt"], args["model"], frame_box, args))

        results = [future.result() for future in futures]

    # Combine results from all sections
    # Combine results from all boxes
    combined_results = combine_results(results, num_rows, num_cols, box_height, box_width)
    matrix = create_zero_matrix(w, h)
    # Visualize the results on the frame

    points = []
    count = 0
    for (x_min, y_min, x_max, y_max, _, class_id) in combined_results:
        if class_id == 15:
            count += 1
            points.extend([(x_min, y_min), (x_max, y_max)])  # Add the corner points

    # Step 2: Create a convex hull
    if points:
        points = np.array(points)
        hull = cv2.convexHull(points)

        # Step 3: Draw the convex hull
        cv2.polylines(frame, [hull], True, (0, 0, 255), 2)
        matrix = cv2.fillConvexPoly(matrix, hull, sinusoidal_scoring(count))
    """
    for (startX, startY, endX, endY, confidence, class_id) in combined_results:
        if class_id == 15:
            label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)
    """
    precision = 1  # e.g., 4 decimal places
    # Save the dense matrix to a text file with reduced precision
    np.savetxt('matrix_reduced_precision.txt', matrix, fmt=f'%.{precision}f')


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    fps.update()

fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

vs.release()
cv2.destroyAllWindows()
