import cv2
import numpy as np
from pupil_apriltags import Detector

# Start camera
cap = cv2.VideoCapture(0)

# Initialize AprilTag detector
apriltag_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Convert image formats
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -----------------------------
    # Steel Ball Detection
    # -----------------------------
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=18,
        minRadius=5,
        maxRadius=20
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(frame, (x, y), r, (0,255,0), 2)
            cv2.circle(frame, (x, y), 2, (0,0,255), 3)
            cv2.putText(frame, "Steel Ball",
                        (x - 30, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)

    # -----------------------------
    # Ping Pong Ball Detection
    # -----------------------------
    lower_orange = np.array([5,120,120])
    upper_orange = np.array([20,255,255])

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    contours_orange, _ = cv2.findContours(mask_orange,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_orange:
        area = cv2.contourArea(cnt)

        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            cv2.circle(frame, (int(x), int(y)), int(radius), (255,0,0), 2)
            cv2.putText(frame, "Ping Pong Ball",
                        (int(x) - 40, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,0,0),
                        2)

    # -----------------------------
    # Robot Detection (30cm cuboids)
    # -----------------------------
    edges = cv2.Canny(gray, 50, 150)

    contours_robot, _ = cv2.findContours(edges,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_robot:

        area = cv2.contourArea(cnt)

        if area > 2000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:

                x, y, w, h = cv2.boundingRect(approx)

                if 80 < w < 400 and 80 < h < 400:

                    cv2.rectangle(frame,
                                  (x, y),
                                  (x+w, y+h),
                                  (0,255,255),
                                  2)

                    cv2.putText(frame,
                                "Robot",
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,255),
                                2)

    # -----------------------------
    # AprilTag Detection
    # -----------------------------
    tags = apriltag_detector.detect(gray)

    for tag in tags:

        corners = tag.corners.astype(int)

        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, pt1, pt2, (0,255,255), 2)

        center = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(frame, center, 5, (0,0,255), -1)

        cv2.putText(frame,
                    f"AprilTag {tag.tag_id}",
                    (center[0] - 40, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2)

    # -----------------------------
    # Display result
    # -----------------------------
    cv2.imshow("Robot + Ball + AprilTag Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()