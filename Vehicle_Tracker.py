import cv2

video = cv2.VideoCapture("video3.mp4")

while True:
    load_frame, frame = video.read()
    if not load_frame:
        break
    cv2.imshow("Vehicle Tracker", frame)
    cv2.waitKey(30)

cv2.waitKey(0)
video.release()