import cv2
# from google.colab.patches import cv2_imshow
import math
import numpy as np
import mediapipe as mp


def video_to_tensor(video_name: str) -> np.ndarray:

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()
    count = 1
    L = [image]

    while success:
        success, image = vidcap.read()
        count += 1
        L.append(image)
    L = L[count//2-32:count//2+32]
    T = np.empty((0), float)
    fail = 0
    for i in L:
        with mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5,
                model_complexity=2) as pose:
            results = pose.process(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                fail += 1
                continue
            x = np.abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x -
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)/2
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + \
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            T = np.hstack((T, temp))
            x = np.abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)/2
            y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)/2
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
            x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            z = 2.7
            temp = np.array([x, y, z], dtype=float)
            T = np.hstack((T, temp))
    temp = np.array([0, 0, 0], dtype=float)
    fail1 = fail//2
    fail2 = fail - fail1
    for i in range(fail1):
        for _ in range(20):
            T = np.hstack((T, temp))
    for i in range(fail2):
        for _ in range(20):
            T = np.hstack((temp, T))
    T = T.reshape(1280, 3)
    return T
