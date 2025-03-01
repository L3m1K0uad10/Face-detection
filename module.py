import time

import cv2 
import mediapipe as mp



class FaceDetector():
    def __init__(self, min_detection_confidence = 0.5):
        self.min_detection_confidence = min_detection_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_confidence) 

    def find_faces(self, img, draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.face_detection.process(img_rgb) 
        bounding_boxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bounding_boxC = detection.location_data.relative_bounding_box
                img_height, img_width, img_channel = img.shape
                bounding_box = int(bounding_boxC.xmin * img_width), int(bounding_boxC.ymin * img_height), \
                            int(bounding_boxC.width * img_width), int(bounding_boxC.height * img_height)
                
                bounding_boxes.append([id, bounding_box, detection.score[0]])
                
                if draw:
                    img = self.fancy_draw(img, bounding_box)

                    cv2.putText(
                        img, 
                        f'SCORE: {int(detection.score[0] * 100)}', 
                        (bounding_box[0], bounding_box[1] - 20), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        6, 
                        (0, 255, 0), 
                        2
                    )

        return img, bounding_boxes
    
    def fancy_draw(self, img, bounding_box, length = 30, thickness = 15, thickness_rect = 2):
        x, y, w, h = bounding_box 
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bounding_box, (255, 0, 255), thickness_rect) 
        # Top left x, y
        cv2.line(img, (x, y), (x + length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + length), (255, 0, 255), thickness)
        # Top right x, y
        cv2.line(img, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness)
        # Bottom left x, y
        cv2.line(img, (x, y1), (x + length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness)
        # Bottom right x, y
        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)

        return img


def main():
    cap = cv2.VideoCapture("Face-Detection/videos/2.mp4")
    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bounding_boxes = detector.find_faces(img, True)
        print(bounding_boxes)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 160), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 6)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()