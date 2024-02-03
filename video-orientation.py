import cv2
import numpy as np
import os

folder = os.path.dirname(os.path.abspath(__file__))
media_folder = os.path.join(folder, "media")


def detect_and_describe(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black = np.zeros_like(image)
    cont = cv2.drawContours(black, contours, -1, (0, 255, 0), 4)

    # Initialize the feature detector (SIFT or ORB)
    detector = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=2, sigma=3)
    # detector = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    # Initialize a matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors of the two images
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by their distances
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def overlay_images(base_image, video_frame, matches, keypoints1, keypoints2):
    # Get the bounding box of the matched region
    base_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(
        -1, 1, 2
    )
    video_pts = np.float32(
        [keypoints2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(video_pts, base_pts, cv2.RANSAC, 5.0)
    h, w = video_frame.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(base_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Describe the transformation
    displacement = (M[0, 2], M[1, 2])
    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    print("Displacement (dx, dy):", displacement)
    print("Scale:", scale)
    print("Rotation (degrees):", rotation)


# def overlay_images(
#     base_image, video_frame, matches, keypoints1, keypoints2, frame_count
# ):
#     # Get the bounding box of the matched region
#     base_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(
#         -1, 1, 2
#     )
#     video_pts = np.float32(
#         [keypoints2[match.trainIdx].pt for match in matches]
#     ).reshape(-1, 1, 2)
#     M, _ = cv2.findHomography(video_pts, base_pts, cv2.RANSAC, 5.0)
#     h, w = video_frame.shape[:2]
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, M)

#     if frame_count % 10 == 0:
#         cv2.polylines(base_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

#     # Describe the transformation
#     displacement = (M[0, 2], M[1, 2])
#     scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
#     rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

#     print("Displacement (dx, dy):", displacement)
#     print("Scale:", scale)
#     print("Rotation (degrees):", rotation)

#     return displacement, scale, rotation


def display_video_overlay(base_image_path, video_path):
    # Load the base image
    base_image = cv2.imread(base_image_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame of the video
    ret, video_frame = video.read()

    # Detect and describe features in the base image
    keypoints1, descriptors1 = detect_and_describe(base_image)

    frame_count = 0

    while ret:
        # Detect and describe features in the current video frame
        keypoints2, descriptors2 = detect_and_describe(video_frame)

        # Match features between the base image and the video frame
        matches = match_features(descriptors1, descriptors2)

        # Overlay the video frame onto the base image
        overlay_images(base_image, video_frame, matches, keypoints1, keypoints2)

        # Draw matched keypoints
        matched_image = cv2.drawMatches(
            base_image, keypoints1, video_frame, keypoints2, matches[:10], None
        )

        # Display the matched keypoints
        cv2.imshow("Large Frame vs. Orientation Detection", matched_image)

        # save the cv2 image
        # cv2.imwrite(os.path.join(media_folder, f"out{frame_count}.jpg"), matched_image)

        # Read the next frame of the video
        ret, video_frame = video.read()

        frame_count += 1

        # Clear polylines every 8 frames
        if frame_count % 4 == 0:
            base_image = cv2.imread(base_image_path)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Store paths of the base image and the video
    base_image_path = os.path.join(media_folder, "cmu-crop1.png")
    video_path = os.path.join(media_folder, "high-alt-trans.mp4")
    # Display the video overlay
    display_video_overlay(base_image_path, video_path)


# def display_video_overlay(base_image_path, video_path):
#     # Load the large aerial shot
#     base_image = cv2.imread(base_image_path)

#     # Open the video file
#     video = cv2.VideoCapture(video_path)

#     # Read the first frame of the video
#     ret, video_frame = video.read()

#     # Detect and describe features in the base image
#     keypoints1, descriptors1 = detect_and_describe(base_image)

#     frame_count = 0

#     while ret:
#         # Detect and describe features in the current video frame
#         keypoints2, descriptors2 = detect_and_describe(video_frame)

#         # Match features between the base image and the video frame
#         matches = match_features(descriptors1, descriptors2)

#         # Overlay the video frame onto the base image
#         overlay_images(
#             base_image, video_frame, matches, keypoints1, keypoints2, frame_count
#         )

#         # Draw matched keypoints
#         matched_image = cv2.drawMatches(
#             base_image, keypoints1, video_frame, keypoints2, matches[:10], None
#         )

#         # Display the matched keypoints
#         cv2.imshow("Matched Keypoints", matched_image)

#         # Read the next frame of the video
#         ret, video_frame = video.read()

#         frame_count += 1

#         # Exit if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     # Release the video capture object and close all windows
#     video.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     # store paths of the base image and the video
#     base_image_path = os.path.join(media_folder, "cmu-crop1.png")
#     video_path = os.path.join(media_folder, "high-alt-trans.mp4")
#     # display the video overlay
#     display_video_overlay(base_image_path, video_path)
