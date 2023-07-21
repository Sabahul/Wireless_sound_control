import cv2
import mediapipe as mp
import math
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Constants
MIN_LENGTH = 50
MAX_LENGTH = 300
MIN_VOLUME = -21
MAX_VOLUME = 0
CIRCLE_RADIUS = 15
LINE_THICKNESS = 3
FONT_SIZE = 1
FONT_THICKNESS = 2
FONT_COLOR = (255, 255, 255)
VOLUME_BAR_COLOR_LOW = (0, 255, 0)
VOLUME_BAR_COLOR_HIGH = (0, 0, 255)
VOLUME_BAR_COLOR_MUTE = (0, 0, 0)
HAND_ICON_RADIUS = 30
HAND_ICON_THICKNESS = 2
HAND_ICON_COLOR = (255, 255, 255)
CONFIDENCE_THRESHOLD = 0.7
PALM_DETECTION_REGION = 50

def calculate_volume(length):
    """Interpolate length to volume within the given range."""
    return np.interp(length, [MIN_LENGTH, MAX_LENGTH], [MIN_VOLUME, MAX_VOLUME])

def calculate_volume_bar(length):
    """Interpolate length to volume bar position within the given range."""
    return np.interp(length, [MIN_LENGTH, MAX_LENGTH], [600, 10])

def calculate_volume_percentage(length):
    """Interpolate length to volume percentage within the given range."""
    return np.interp(length, [MIN_LENGTH, MAX_LENGTH], [0, 100])

def process_hand_landmarks(hand_landmarks, img, volume):
    """Process hand landmarks and update volume accordingly."""
    lmList = []
    for id, lm in enumerate(hand_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])

    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x1, y1), CIRCLE_RADIUS, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), CIRCLE_RADIUS, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), LINE_THICKNESS)

        z1, z2 = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < MIN_LENGTH:
            cv2.circle(img, (z1, z2), CIRCLE_RADIUS, (255, 255, 255), cv2.FILLED)

        try:
            volume_level = calculate_volume(length)
            volBar = calculate_volume_bar(length)
            volPer = calculate_volume_percentage(length)

            volume.SetMasterVolumeLevel(volume_level, None)
            volume_mute = volume.GetMute()

            # Change the color of the volume bar based on the volume level
            if volume_mute:
                vol_color = VOLUME_BAR_COLOR_MUTE
            elif volume_level < 0:
                vol_color = VOLUME_BAR_COLOR_LOW
            else:
                vol_color = VOLUME_BAR_COLOR_HIGH

            cv2.rectangle(img, (50, 150), (85, 400), vol_color, LINE_THICKNESS)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), vol_color, cv2.FILLED)
            cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

            # Display instructions on how to control the volume
            instructions1 = "Spread your fingers to adjust volume."
            instructions2 = "Bring fingertips close to mute."
            cv2.putText(img, instructions1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FONT_COLOR, 2)
            cv2.putText(img, instructions2, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FONT_COLOR, 2)

            # Visual feedback for volume changes
            if abs(volume_level) > 1.0:
                volume_change_color = (0, 255, 0) if volume_level > 0 else (0, 0, 255)
                cv2.circle(img, (cx, cy), CIRCLE_RADIUS, volume_change_color, LINE_THICKNESS)

        except Exception as e:
            print(f"Error updating volume: {e}")

    return length  # Return the length value

def display_volume_indicator(img, volume_level):
    """Display the current volume level as a visual indicator."""
    volume_indicator_x = 40
    volume_indicator_y = int(np.interp(volume_level, [MIN_VOLUME, MAX_VOLUME], [400, 150]))
    cv2.circle(img, (volume_indicator_x, volume_indicator_y), HAND_ICON_RADIUS, HAND_ICON_COLOR, HAND_ICON_THICKNESS)

def display_volume_message(img, volume_level):
    """Display a friendly message based on the current volume level."""
    if volume_level > -5:
        message = "Volume is high! Lower it down a bit."
    elif volume_level > -12:
        message = "Volume is moderate. Enjoy your music!"
    else:
        message = "Volume is low. Turn it up for a better experience!"
    cv2.putText(img, message, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_ICON_COLOR, 2)

def display_hand_icon(img, hand_landmarks):
    """Display an animated hand icon that changes based on the hand gesture detected."""
    # ... (Your code to display the animated hand icon goes here)

def main():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=CONFIDENCE_THRESHOLD, min_tracking_confidence=CONFIDENCE_THRESHOLD)
    mpDraw = mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Initialize length and handLms with default values
    length = MAX_LENGTH
    handLms = None

    try:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                # Get the first detected hand (assumes only one hand in the frame)
                handLms = results.multi_hand_landmarks[0]

                # Draw hand landmarks on the image
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                # Process hand landmarks and update volume
                length = process_hand_landmarks(handLms, img, volume)

            # Calculate volume level and display volume indicator/message/hand icon
            volume_level = calculate_volume(length)
            display_volume_indicator(img, volume_level)
            display_volume_message(img, volume_level)

            if handLms:
                display_hand_icon(img, handLms)  # Display the hand icon if hand landmarks are detected

            cv2.imshow("Hand Gesture Volume Control", img)
            key = cv2.waitKey(1)
            if key == 27:  # Press 'Esc' key to exit the loop
                break

    except KeyboardInterrupt:
        print("Exiting gracefully due to user interruption.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Welcome to Hand Gesture Volume Control!")
    print("Adjust your computer's volume by spreading your fingers or bringing fingertips close together.")
    print("Press 'Esc' key to exit the program.")
    main()
