# import packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
import scipy.sparse as sp

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
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#333

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
def combine_results(sections_results, section_height):
    combined_results = []
    for section_idx, results in enumerate(sections_results):
        for (startX, startY, endX, endY, confidence, class_id) in results:
            startY += section_idx * section_height
            endY += section_idx * section_height
            combined_results.append((startX, startY, endX, endY, confidence, class_id))
    return combined_results

# Load the model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("/Users/bkleyner/Desktop/dronefootage.mp4")  # use 0 for webcam, or replace with video file path
fps = FPS().start()

# Process each frame
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=2560)
    (h, w) = frame.shape[:2]
    print(h, w)
    num_sections = 4
    section_height = h // num_sections

    # Process each frame section in parallel
    with ThreadPoolExecutor(max_workers=num_sections) as executor:
        futures = [executor.submit(process_frame_section, args["prototxt"], args["model"], frame[i * section_height:(i + 10) * section_height, :], args)
                   for i in range(num_sections)]
        results = [future.result() for future in futures]

    # Combine results from all sections
    combined_results = combine_results(results, section_height)
    matrix = create_zero_matrix(w, h)
    # Visualize the results on the frame
    for (startX, startY, endX, endY, confidence, class_id) in combined_results:
        label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)
        matrix = fill_matrix_with_value(matrix, startX, startY, endX, endY, confidence)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)

    # Define the precision for floating-point numbers
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
