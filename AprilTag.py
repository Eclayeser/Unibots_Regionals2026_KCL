import cv2
import numpy as np
from pupil_apriltags import Detector

TARGET_TAGS = [1, 2]

cap = cv2.VideoCapture(0)

detector = Detector(families="tag36h11")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(gray)

    frame_center = frame.shape[1] // 2

    target = None

    for d in detections:
        if d.tag_id in TARGET_TAGS:
            target = d
            break

    if target is not None:

        corners = target.corners.astype(int)

        for i in range(4):
            cv2.line(frame,
                     tuple(corners[i]),
                     tuple(corners[(i+1)%4]),
                     (0,255,0),2)

        center = (int(target.center[0]), int(target.center[1]))
        cv2.circle(frame, center, 5, (0,0,255), -1)

        cv2.putText(frame,
                    f"Tag {target.tag_id}",
                    (center[0]-30, center[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2)

        offset = center[0] - frame_center

        if offset < -40:
            print("TURN LEFT")

        elif offset > 40:
            print("TURN RIGHT")

        else:
            print("MOVE FORWARD")

    else:
        print("SEARCHING")

    cv2.imshow("AprilTag Lock", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

