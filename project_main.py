# Import the libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

import threading
import signal
import time
import copy

# global variable used to watch out for keyboard interrupt.
# safely exit threading
kill_threads = False
def signal_handler(signal, frame):
    global kill_threads
    print('Interrupted!')
    kill_threads = True

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
# Initialize mediapipe model
hands = mpHands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
# Load gesture recognizer model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()


def prediction_thread():
    global handslms
    global className
    # delay thread to let the video start
    time.sleep(2)

    while(True):
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, y, c = framergb.shape
        # Get hand landmark prediction
        result = hands.process(framergb)
        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx,lmy])
            # Predict gesture in Hand Gesture Recognition project
            prediction = model.predict([landmarks], verbose=0)
            classID = np.argmax(prediction)
            className = classNames[classID]


def main():
    # if a variable is declared as global it means "this function is using a global 
    # variable that can be written and read anywhere". Variables declared outside of 
    # a function but that aren't declared global can only be read

    # I am declaring global variables so that other threads can write to these variables
    global frame
    global cap
    # initialize signal interrupt catcher
    signal.signal(signal.SIGINT, signal_handler)
    # Initialize the webcam for Hand Gesture Recognition Python project
    cap = cv2.VideoCapture(0)
    # define thread that deals with making calculations
    prediction_thread_handler = threading.Thread(target=prediction_thread)
    # start thread
    prediction_thread_handler.start()

    while(kill_threads == False):
        # Read each frame from the camera
        _, frame_temp = cap.read()
        # Flip the frame vertically
        frame = cv2.flip(frame_temp, 1)

        frame_post = copy.deepcopy(frame)

        try:
            mpDraw.draw_landmarks(frame_post, handslms, 
                mpHands.HAND_CONNECTIONS)
            # show the prediction on the frame
            cv2.putText(frame_post, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2, cv2.LINE_AA)
        except:
            pass
        # Show the final output
        cv2.imshow("Output", frame_post)
        cv2.waitKey(1) & 0xFF
        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()
    prediction_thread_handler.join()



if __name__=="__main__":
    main()