from ultralytics import YOLO
import cv2
import cvzone
import math

video = cv2.VideoCapture("../video/video_1.mp4")

model = YOLO("../runs/detect/train/weights/best.pt")
model = YOLO("../Yolo_init/yolov8n.pt")

classNames = ["mehedi"]

while True:
    success, img = video.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))

            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]}: {conf}%',(max(0,x1),max(30,y1)),scale=1.5,thickness=2)

    cv2.waitKey(1)
    cv2.imshow("VideoPlay", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

