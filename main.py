
"""
AI-Based Hand Tracking for Robotic Arm Control
Webcam simulation only. No physical robotic arm is required.

Controls:
- Move hand left/right: rotate shoulder.
- Move hand up/down: control elbow.
- Pinch thumb and index finger: close gripper.
- Separate thumb and index finger: open gripper.
- Press Q or ESC to quit.
"""

import math
import cv2
import mediapipe as mp


# -------------------- Settings --------------------
CAMERA_INDEX = 0
WINDOW_NAME = "AI Hand Tracking - Virtual Robotic Arm"

ARM_LENGTH_1 = 150
ARM_LENGTH_2 = 120
GRIPPER_OPEN_DISTANCE = 55
SMOOTHING = 0.25


def clamp(value, low, high):
    return max(low, min(high, value))


def draw_text(image, text, position, scale=0.6,
              color=(255, 255, 255), thickness=2):
    cv2.putText(
        image, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def draw_joint(image, point, radius=10,
               color=(0, 220, 255)):
    cv2.circle(image, point, radius, color, -1,
               cv2.LINE_AA)
    cv2.circle(image, point, radius, (30, 30, 30), 2,
               cv2.LINE_AA)


def draw_virtual_arm(image, shoulder_angle_deg,
                     elbow_angle_deg, gripper_open):
    """Draw a two-link virtual robotic arm."""
    h, w = image.shape[:2]
    base = (int(w * 0.76), int(h * 0.78))

    a1 = math.radians(shoulder_angle_deg)
    a2 = math.radians(elbow_angle_deg)

    elbow = (
        int(base[0] + ARM_LENGTH_1 * math.cos(a1)),
        int(base[1] - ARM_LENGTH_1 * math.sin(a1)),
    )

    wrist_angle = a1 + a2
    wrist = (
        int(elbow[0] + ARM_LENGTH_2 * math.cos(wrist_angle)),
        int(elbow[1] - ARM_LENGTH_2 * math.sin(wrist_angle)),
    )

    # Base, arm links and joints
    cv2.circle(image, base, 27, (80, 80, 80), -1,
               cv2.LINE_AA)
    cv2.circle(image, base, 27, (230, 230, 230), 2,
               cv2.LINE_AA)

    cv2.line(image, base, elbow, (255, 160, 40), 18,
             cv2.LINE_AA)
    cv2.line(image, elbow, wrist, (70, 200, 255), 14,
             cv2.LINE_AA)

    draw_joint(image, base, 13, (180, 180, 180))
    draw_joint(image, elbow, 12, (0, 180, 255))
    draw_joint(image, wrist, 10, (0, 255, 120))

    # Gripper fingers
    gap = 18 if gripper_open else 5
    finger_len = 35

    perp_x = -math.sin(wrist_angle)
    perp_y = -math.cos(wrist_angle)
    forward_x = math.cos(wrist_angle)
    forward_y = -math.sin(wrist_angle)

    for sign in (-1, 1):
        start = (
            int(wrist[0] + sign * gap * perp_x),
            int(wrist[1] + sign * gap * perp_y),
        )

        end = (
            int(start[0] + finger_len * forward_x
                + sign * 15 * perp_x),
            int(start[1] + finger_len * forward_y
                + sign * 15 * perp_y),
        )

        cv2.line(image, start, end, (0, 255, 120), 6,
                 cv2.LINE_AA)

    # Arm status
    x = int(w * 0.59)
    draw_text(image, "VIRTUAL ROBOTIC ARM",
              (x, 35), 0.7, (0, 255, 120), 2)
    draw_text(image,
              f"Shoulder: {int(shoulder_angle_deg)} deg",
              (x, 65))
    draw_text(image,
              f"Elbow: {int(elbow_angle_deg)} deg",
              (x, 92))

    gripper_color = (
        (0, 255, 120) if gripper_open else (0, 120, 255)
    )
    gripper_status = "OPEN" if gripper_open else "CLOSED"

    draw_text(image, f"Gripper: {gripper_status}",
              (x, 119), 0.6, gripper_color)


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(
            "ERROR: Webcam could not be opened. "
            "Check camera permissions or CAMERA_INDEX."
        )
        return

    shoulder_target = 90.0
    elbow_target = 45.0

    shoulder_smooth = shoulder_target
    elbow_smooth = elbow_target
    gripper_open = True

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:

            print("Project started!")
            print("Move your hand to control the virtual arm.")
            print("Pinch thumb and index finger to close gripper.")
            print("Press Q or ESC to quit.")

            while True:
                ok, frame = cap.read()

                if not ok:
                    print("ERROR: Could not read webcam frame.")
                    break

                # Mirror camera for natural movement
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Right-side simulation panel
                panel_x = int(w * 0.55)
                overlay = frame.copy()

                cv2.rectangle(
                    overlay, (panel_x, 0), (w, h),
                    (20, 25, 32), -1
                )
                frame = cv2.addWeighted(
                    overlay, 0.78, frame, 0.22, 0
                )

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    # Hand landmarks
                    wrist = hand.landmark[0]
                    thumb_tip = hand.landmark[4]
                    index_tip = hand.landmark[8]

                    palm_x = wrist.x
                    palm_y = wrist.y

                    # Convert hand position to joint angles
                    shoulder_target = clamp(
                        180 * (1 - palm_x), 15, 165
                    )
                    elbow_target = clamp(
                        150 * (1 - palm_y), 10, 140
                    )

                    # Smooth joint movements
                    shoulder_smooth += SMOOTHING * (
                        shoulder_target - shoulder_smooth
                    )
                    elbow_smooth += SMOOTHING * (
                        elbow_target - elbow_smooth
                    )

                    # Thumb-index distance in approximate pixels
                    dx = (thumb_tip.x - index_tip.x) * w
                    dy = (thumb_tip.y - index_tip.y) * h
                    pinch_distance = math.hypot(dx, dy)

                    gripper_open = (
                        pinch_distance >= GRIPPER_OPEN_DISTANCE
                    )

                    hand_px = int(palm_x * w)
                    hand_py = int(palm_y * h)

                    cv2.circle(
                        frame, (hand_px, hand_py), 10,
                        (0, 255, 255), -1
                    )
                    draw_text(
                        frame, "HAND DETECTED", (20, 35),
                        0.7, (0, 255, 0), 2
                    )
                    draw_text(
                        frame,
                        f"Pinch distance: {int(pinch_distance)} px",
                        (20, 65)
                    )

                else:
                    draw_text(
                        frame, "Show one hand to the camera",
                        (20, 35), 0.7, (0, 180, 255), 2
                    )
                    draw_text(
                        frame, "Arm holds its last position",
                        (20, 65), 0.55
                    )

                draw_virtual_arm(
                    frame, shoulder_smooth,
                    elbow_smooth, gripper_open
                )

                # Instructions panel
                draw_text(
                    frame, "CONTROLS", (20, h - 105),
                    0.65, (255, 255, 255), 2
                )
                draw_text(
                    frame, "Move hand left/right: shoulder",
                    (20, h - 78), 0.5
                )
                draw_text(
                    frame, "Move hand up/down: elbow",
                    (20, h - 54), 0.5
                )
                draw_text(
                    frame, "Pinch thumb + index: close gripper",
                    (20, h - 30), 0.5
                )
                draw_text(
                    frame, "Press Q or ESC to quit",
                    (int(w * 0.59), h - 20), 0.5
                )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
