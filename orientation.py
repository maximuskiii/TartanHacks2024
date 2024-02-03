import cv2
import numpy as np
import os


folder = os.path.dirname(os.path.abspath(__file__))
media_folder = os.path.join(folder, "media")


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(contrastThreshold=0.07, edgeThreshold=8)
    # detector = cv2.ORB_create()
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


# WORKS but has a lot of noise
def overlay_images(large_image, small_image, matches, keypoints1, keypoints2):
    large_pts = np.float32(
        [keypoints1[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    small_pts = np.float32(
        [keypoints2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(small_pts, large_pts, cv2.RANSAC, 5.0)
    h, w = small_image.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(large_image, [np.int32(dst)], True, (50, 50, 250), 3, cv2.LINE_AA)

    # displacement information
    displacement = (M[0, 2], M[1, 2])
    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    return displacement, scale, rotation


def display_matched_keypoints(large_image_path, small_image_path):
    large_image = cv2.imread(large_image_path)
    small_image = cv2.imread(small_image_path)
    keypoints1, descriptors1 = detect_and_describe(large_image)
    keypoints2, descriptors2 = detect_and_describe(small_image)
    matches = match_features(descriptors1, descriptors2)

    disp, scale, rot = overlay_images(
        large_image, small_image, matches, keypoints1, keypoints2
    )

    matched_image = cv2.drawMatches(
        large_image, keypoints1, small_image, keypoints2, matches[:10], None
    )

    cv2.imshow("Matched Keypoints", matched_image)
    cv2.waitKey(0)

    # print the displacement, scale, and rotation
    print("Displacement: ", disp)
    print("Scale: ", scale)
    print("Rotation: ", rot)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    large = os.path.join(media_folder, "cmu-crop1.png")
    small = os.path.join(media_folder, "ha.png")
    display_matched_keypoints(large, small)
