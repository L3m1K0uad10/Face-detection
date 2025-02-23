import time

import cv2 
import mediapipe as mp



cap = cv2.VideoCapture("Face-Detection/videos/1.mp4")
pTime = 0


mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection() # initializing the face detection model

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting the image to RGB
    results = face_detection.process(img_rgb) # processing the image
    #print(results)

    # extracting information from the results
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mp_draw.draw_detection(img, detection) # drawing the bounding box by using the mediapipe draw function
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bounding_boxC = detection.location_data.relative_bounding_box
            img_height, img_width, img_channel = img.shape
            bounding_box = int(bounding_boxC.xmin * img_width), int(bounding_boxC.ymin * img_height), \
                           int(bounding_boxC.width * img_width), int(bounding_boxC.height * img_height)
            
            cv2.rectangle(img, bounding_box, (255, 0, 255), 2) # drawing manually the bounding box by using the opencv rectangle function
            cv2.putText(
                img, 
                f'SCORE: {int(detection.score[0] * 100)}', 
                (bounding_box[0], bounding_box[1] - 20), 
                cv2.FONT_HERSHEY_PLAIN, 
                6, 
                (0, 255, 0), 
                2
            )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 160), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 6)

    cv2.imshow("Image", img)
    cv2.waitKey(1)