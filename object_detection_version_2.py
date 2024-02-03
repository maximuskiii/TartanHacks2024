import math
import threading

from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
from concurrent.futures import ThreadPoolExecutor




def sinusoidal_scoring(num_people):
    if num_people <= 0:
        return 0

    amplitude = 5
    frequency = 0.5
    log_base = 10

    score = amplitude * math.sin(frequency * math.log(num_people, log_base))
    return score

def create_zero_matrix(n, m):
    return np.zeros((n, m))

def fill_matrix_with_value(matrix, start_x, start_y, end_x, end_y, value):
    n = matrix.shape[0]
    matrix[start_x:end_x, start_y:end_y] = value
    return matrix

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


def combine_results(sections_results, num_rows, num_cols, box_height, box_width):
    combined_results = []
    for row in range(num_rows):
        for col in range(num_cols):
            start_y = row * box_height
            start_x = col * box_width
            for detection in sections_results[row * num_cols + col]:
                x_min, y_min, x_max, y_max, confidence, class_id = detection
                x_min += start_x
                x_max += start_x
                y_min += start_y
                y_max += start_y

                combined_results.append((x_min, y_min, x_max, y_max, confidence, class_id))

    return combined_results
def run_models_protest():
    ap = argparse.ArgumentParser()
    path_to_model = "SSD_MobileNet.caffemodel"
    path_to_prototxt = "SSD_MobileNet_prototxt.txt"
    ap.add_argument("-c", "--confidence", type=float, default=0.6,
                    help="minimum probability to filter weak predictions")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (192, 192, 192),
        (128, 0, 0),
        (128, 128, 0),
        (0, 128, 0),
        (128, 0, 128),
        (105, 105, 105),
        (0, 0, 128),
        (255, 165, 0),
        (255, 215, 0),
        (66, 206, 245),
        (0, 100, 0),
        (139, 0, 139),
        (85, 107, 47),
        (255, 69, 0),
        (75, 0, 130)
    ]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(path_to_prototxt, path_to_model)


    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture("/Users/bkleyner/Desktop/dronefootage.mp4")
    fps = FPS().start()
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=2160)

        (h, w) = frame.shape[:2]
        num_rows, num_cols = 2, 3
        box_height, box_width = h // num_rows, w // num_cols

        with ThreadPoolExecutor(max_workers=num_rows * num_cols) as executor:
            futures = []
            for row in range(num_rows):
                for col in range(num_cols):
                    start_y = row * box_height
                    end_y = (row + 1) * box_height
                    start_x = col * box_width
                    end_x = (col + 1) * box_width
                    frame_box = frame[start_y:end_y, start_x:end_x]
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    futures.append(executor.submit(process_frame_section, path_to_prototxt, path_to_model, frame_box, args))

            results = [future.result() for future in futures]

        combined_results = combine_results(results, num_rows, num_cols, box_height, box_width)
        matrix = create_zero_matrix(h, w)

        points = []
        count = 0
        for (x_min, y_min, x_max, y_max, _, class_id) in combined_results:
            if class_id == 15:
                count += 1
                points.extend([(x_min, y_min), (x_max, y_max)])

        if points:
            points = np.array(points)
            hull = cv2.convexHull(points)

            cv2.polylines(frame, [hull], True, (0, 0, 255), 2)
            matrix = cv2.fillConvexPoly(matrix, hull, sinusoidal_scoring(count))

        for (startX, startY, endX, endY, confidence, class_id) in combined_results:
            if class_id == 15:
                label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()

    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()
    cv2.destroyAllWindows()

def run_models_other():
    ap = argparse.ArgumentParser()
    path_to_model = "SSD_MobileNet.caffemodel"
    path_to_prototxt = "SSD_MobileNet_prototxt.txt"
    ap.add_argument("-c", "--confidence", type=float, default=0.6,
                    help="minimum probability to filter weak predictions")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (192, 192, 192),
        (128, 0, 0),
        (128, 128, 0),
        (0, 128, 0),
        (128, 0, 128),
        (105, 105, 105),
        (0, 0, 128),
        (255, 165, 0),
        (255, 215, 0),
        (66, 206, 245),
        (0, 100, 0),
        (139, 0, 139),
        (85, 107, 47),
        (255, 69, 0),
        (75, 0, 130)
    ]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(path_to_prototxt, path_to_model)


    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture("/Users/bkleyner/Desktop/test4.mp4")
    fps = FPS().start()
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=2160)

        (h, w) = frame.shape[:2]
        num_rows, num_cols = 2, 3
        box_height, box_width = h // num_rows, w // num_cols

        with ThreadPoolExecutor(max_workers=num_rows * num_cols) as executor:
            futures = []
            for row in range(num_rows):
                for col in range(num_cols):
                    start_y = row * box_height
                    end_y = (row + 1) * box_height
                    start_x = col * box_width
                    end_x = (col + 1) * box_width
                    frame_box = frame[start_y:end_y, start_x:end_x]
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    futures.append(executor.submit(process_frame_section, path_to_prototxt, path_to_model, frame_box, args))

            results = [future.result() for future in futures]

        combined_results = combine_results(results, num_rows, num_cols, box_height, box_width)
        matrix = create_zero_matrix(h, w)

        points = []
        count = 0
        for (x_min, y_min, x_max, y_max, _, class_id) in combined_results:
            if class_id == 15:
                count += 1
                points.extend([(x_min, y_min), (x_max, y_max)])

        if points:
            points = np.array(points)
            hull = cv2.convexHull(points)

            cv2.polylines(frame, [hull], True, (0, 0, 255), 2)
            matrix = cv2.fillConvexPoly(matrix, hull, sinusoidal_scoring(count))

        for (startX, startY, endX, endY, confidence, class_id) in combined_results:
            if class_id == 15:
                label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()

    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()
    cv2.destroyAllWindows()

thread1 = threading.Thread(target=run_models_protest())
thread2 = threading.Thread(target=run_models_other())

thread1.start()
thread2.start()

thread1.join()
thread2.join()
