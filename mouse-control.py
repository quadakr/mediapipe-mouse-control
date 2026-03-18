import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

cap = cv2.VideoCapture(0)
dotool = subprocess.Popen(["sudo", "dotool"], stdin=subprocess.PIPE, text=True)


smooth_alpha = 0.87
input_scale = 0.25
offset_x = -0.65
offset_y = -0.7
scroll_speed = 80

moving = False

smoothed_x = None
smoothed_y = None

dragging_left_index = False
dragging_right_index = False

dragging_left_pinky = False
dragging_right_pinky = False

dragging_left_middle = False
dragging_right_middle = False

dragging_left_ring = False
dragging_right_ring = False

last_scroll_y = None
last_ring_state = False

x_old = 1
y_old = 1


def detect_pinch(positions, finger, sens):
    thumb = np.array(positions["thumb"])
    tip = np.array(positions[finger])
    wrist = np.array(positions["wrist"])
    index = np.array(positions["index"])

    distance = np.linalg.norm(thumb - tip)
    hand_size = np.linalg.norm(wrist - index)

    if hand_size > 0:
        return (distance / hand_size) < sens
    return False


def smooth_norm(x, y):
    global smoothed_x, smoothed_y

    if smoothed_x is None:
        smoothed_x, smoothed_y = x, y
        return x, y

    smoothed_x = smoothed_x * smooth_alpha + x * (1 - smooth_alpha)
    # smoothed_y = smoothed_y * smooth_alpha + y * (1 - smooth_alpha)

    return smoothed_x, smoothed_y


def rescale_input(x, y):
    half = input_scale / 2

    min_x = 0.5 - half
    max_x = 0.5 + half
    min_y = 0.5 - half
    max_y = 0.5 + half

    x = (x - min_x) / (max_x - min_x)
    y = (y - min_y) / (max_y - min_y)

    x += offset_x
    y += offset_y

    return x, y


def move_mouse(x, y):
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    dotool.stdin.write(f"mouseto {x:.4f} {y:.4f}\n")
    dotool.stdin.flush()


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            continue

        lm = results.multi_hand_landmarks[0].landmark

        x_norm = lm[0].x
        y_norm = lm[0].y

        x_norm, y_norm = smooth_norm(x_norm, y_norm)
        x_norm, y_norm = rescale_input(x_norm, y_norm)

        if moving:
            move_mouse(x_norm - ((x_norm - x_old) / 2), y_norm - ((y_norm - y_old) / 2))

        time.sleep(0.015)

        if moving:
            move_mouse(x_norm, y_norm)

        x_old = x_norm
        y_old = y_norm

        h, w, _ = frame.shape
        positions = {
            "thumb": (int(lm[4].x * w), int(lm[4].y * h)),
            "index": (int(lm[8].x * w), int(lm[8].y * h)),
            "middle": (int(lm[12].x * w), int(lm[12].y * h)),
            "ring": (int(lm[16].x * w), int(lm[16].y * h)),
            "pinky": (int(lm[20].x * w), int(lm[20].y * h)),
            "wrist": (int(lm[0].x * w), int(lm[0].y * h)),
        }

        pinch_left_index = detect_pinch(positions, "index", 0.22)
        pinch_left_pinky = detect_pinch(positions, "pinky", 0.22)
        pinch_left_middle = detect_pinch(positions, "middle", 0.2)
        pinch_left_ring = detect_pinch(positions, "ring", 0.27)

        if pinch_left_index and not dragging_left_index:
            dotool.stdin.write("buttondown left\n")
            dotool.stdin.flush()
            dragging_left_index = True
        elif not pinch_left_index and dragging_left_index:
            dotool.stdin.write("buttonup left\n")
            dotool.stdin.flush()
            dragging_left_index = False

        if pinch_left_pinky and not dragging_left_pinky:
            dotool.stdin.write("buttondown right\n")
            dotool.stdin.flush()
            dragging_left_pinky = True
        elif not pinch_left_pinky and dragging_left_pinky:
            dotool.stdin.write("buttonup right\n")
            dotool.stdin.flush()
            dragging_left_pinky = False

        if pinch_left_ring and not dragging_left_ring:
            dragging_left_ring = True
            moving = True
        elif not pinch_left_ring and dragging_left_ring:
            dragging_left_ring = False
            moving = False

        if pinch_left_middle:
            if not dragging_left_middle:
                dragging_left_middle = True
                last_scroll_y = positions["wrist"][1]
            else:
                current_y = positions["wrist"][1]
                delta = last_scroll_y - current_y

                if abs(delta) > 5: 
                    scroll_amount = int(delta / 8) 
                    dotool.stdin.write(f"wheel {scroll_amount}\n")
                    dotool.stdin.flush()
                    last_scroll_y = current_y
        else:
            dragging_left_middle = False
            last_scroll_y = None


except KeyboardInterrupt:
    pass
finally:
    cap.release()
    hands.close()
