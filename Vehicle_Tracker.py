import cv2

#Turns video file into frames, each frame being an image represented by a numpy array
video = cv2.VideoCapture("video3.mp4")

#Stores whether there is another frame to load, and the frame itself
load_frame, frame = video.read()

#Allows you to select ROIT
x, y, width, height = cv2.selectROI(
    "Select a vehicle",
    frame,
    showCrosshair=True,
    fromCenter=False
)

print(x, y, width, height)

vehicle = frame[y:y + height, x:x + width] #Crops image to show only ROI

cv2.imshow("Selected vehicle", vehicle)
cv2.waitKey(0)

"""while True:
    load_frame, frame = video.read()
    if not load_frame:
        break
    cv2.imshow("Vehicle Tracker", frame)
    cv2.waitKey(30)"""

cv2.waitKey(0)
video.release()