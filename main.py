import os
import time
import cv2
import random
import psutil
import wikipedia
import threading
import subprocess
import numpy as np
import pandas as pd
from gtts import gTTS
from time import sleep
import mediapipe as mp
import tensorflow as tf
from mutagen.mp3 import MP3
from chatbot import Chatbot
from collections import Counter
from webdriver_manager.chrome import ChromeDriverManager

#emotion model 
mp_face_detection = mp.solutions.face_detection
interpreter = tf.lite.Interpreter(model_path="/home/mycar1130/quocanh/emoitiondetection/emotion_detection/model/emotion_mobilenet_v2.tflite")
interpreter.allocate_tensors()

# input for emotion model
offset = 0
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
emotions_df = []
col_names = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","Unknown", "NF"]

# using mediapipe for face detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(color=(245,66,230),thickness=1, circle_radius=1)

# create class chatbot
chatbot = Chatbot()

# convert english sentence to mp3
wikipedia.set_lang('en')
language = 'en'
path = ChromeDriverManager().install()

# Default webcam
cap = cv2.VideoCapture(0)

# ================Config=====================
speaking = False
show_lmark = False
width,height = 640,480
# ==========================================


def find_most_frequent_value(values):
    counts = Counter(values)
    most_frequent_value = counts.most_common(1)[0][0]
    return most_frequent_value

def ask(emotion):
    intents_list = chatbot.predict_class(emotion)
    res = chatbot.get_response(intents_list)
    return (str(res))

# Speak function
def speak(emotion):
    message = ask(str(emotion))
    global speaking
    speaking = True
    print("Bot: {}".format(message))
    tts = gTTS(text=message, lang=language, slow=False)
    tts.save("sound.mp3")
    # audio = MP3("sound.mp3")
    # Use mpg123 to play the audio file
    subprocess.call(["mpg123", "sound.mp3"])
    os.remove("sound.mp3")
    sleep(2)
    threading.Thread(target=play_random_song, args=(str(emotion),)).start()

# play random song
def play_random_song(emotion):
    folder_path = '/home/mycar1130/quocanh/resfes/music/'+str(emotion)
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return
    
    audio_files = [file for file in os.listdir(folder_path) if file.endswith(('.mp3', '.wav', '.ogg'))]
    
    if not audio_files:
        print("No audio files found in the folder.")
        return
    
    random_song = random.choice(audio_files)
    song_path = os.path.join(folder_path, random_song)
    # audio = MP3(song_path)
    # Use subprocess to call mpg123 and play the song
    process = subprocess.call(['mpg123', song_path])
    sleep(2)
    global speaking
    speaking = False

# using model to predict emotion
def check_emotion(crop_img):
    img = cv2.resize(crop_img,(48,48))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.repeat(img[..., np.newaxis], 3, -1)

    # cv2.imshow('processed',img)
    img =img / 255
    img = img.astype('float32')

    input_shape = input_details[0]['shape']
    input_tensor= np.expand_dims(img,0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred = np.squeeze(output_data)
    highest_pred_loc = np.argmax(pred)
    return highest_pred_loc

# Save video
# fps =  cap.get(cv2.CAP_PROP_FPS)
# size = (width, height)
# vids = cv2.VideoWriter("emotion_results.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

def main():
    '''
    Detects face and outputs an emotion
    '''
     #===========================
    emotions = []
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                
                if not success:
                    print("Ignoring empty camera frame.")
                    break
                
                image = cv2.resize(image,(640,480))
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
                results1 = face_detection.process(image)
                
                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results1.detections:
                    for detection in results1.detections:
                        location_data = detection.location_data
                        bb = location_data.relative_bounding_box
                        
                        bbox_points = {
                            "xmin" : int(bb.xmin * width),
                            "ymin" : int(bb.ymin * height),
                            "xmax" : int(bb.width * width + bb.xmin * width),
                            "ymax" : int(bb.height * height + bb.ymin * height)
                        }

                        x,y,w,h = bbox_points['xmin'],bbox_points['ymin'],bbox_points['xmax'],bbox_points['ymax']
                        try:
                            cropped_image = image[y-offset:h+ offset,x-offset:w+offset]
                            emo_det = check_emotion(cropped_image)
                            # col_names = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","Unknown", "NF"]
                            if speaking == False: 
                                emotions.append(emo_det)
                            cv2.putText(image,col_names[emo_det],(w,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(0,0,255),2)
                            cv2.rectangle(image,(x,y),(w,h),(0,255,0),2)
                        except:
                            pass
                
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Detect the face landmarks
                results2 = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                global show_lmark
                if results2.multi_face_landmarks and show_lmark == False:
                    for face_landmarks in results2.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())

                print(str(speaking) + " " + str(len(emotions)))
                if (speaking == False and len(emotions) == 10):
                    emotion = find_most_frequent_value(emotions)
                    emotions = []
                    if emotion == 1:
                        t = threading.Thread(target=speak, args=("happiness",))
                        t.start()
                    elif emotion == 2 :
                        t = threading.Thread(target=speak, args=("surprise",))
                        t.start()
                    elif emotion == 3 :
                        t = threading.Thread(target=speak, args=("sadness",))
                        t.start()
                    elif emotion == 4 :
                        t = threading.Thread(target=speak, args=("anger",))
                        t.start()
                    elif emotion == 5 :
                        t = threading.Thread(target=speak, args=("disgust",))
                        t.start()
                    elif emotion == 6 :
                        t = threading.Thread(target=speak, args=("fear",))
                        t.start()
                    elif emotion == 7 :
                        t = threading.Thread(target=speak, args=("contempt",))
                        t.start()
                
                
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    show_lmark = not show_lmark
                
                cv2.imshow('Emotion Recog', image)
                # vids.write(image)
                
                if cv2.waitKey(30) & 0xFF == 27:
                    # Terminate the Python process running main.py
                    subprocess.Popen(["pkill", "-f", "mpg123"])
                    for proc in psutil.process_iter(['cmdline']):
                        if proc.info['cmdline'] and 'main.py' in proc.info['cmdline'][1:]:
                            proc.kill()
                    # Terminate all subprocesses

                    # # Terminate all processes
                    # for proc in psutil.process_iter():
                    #     proc.terminate()

                    # # Terminate all threads
                    # for thread in threading.enumerate():
                    #     if thread != threading.main_thread():
                    #         thread.join(0)
        cap.release()
        # vids.release()

if __name__ == "__main__":
  main()