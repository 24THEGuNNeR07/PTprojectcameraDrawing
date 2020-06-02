import numpy as np
import cv2
from collections import deque

# warunki brzegowe dla kolorów (przedmioty rysujące)
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

greenLower = np.array([0,100,0])
greenUpper = np.array([0,255,0])

redLower = np.array([165,42,42])
redUpper = np.array([255,0,0])

kernel = np.ones((5, 5), np.uint8)

bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

paintWindow = np.zeros((471,636,3)) + 255

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# włączenie kamery
camera = cv2.VideoCapture(0)

# pętla
while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not grabbed:
        break

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    greenMask = cv2.inRange(hsv, greenLower, greenUpper)
    greenMask = cv2.erode(greenMask, kernel, iterations=2)
    greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_OPEN, kernel)
    greenMask = cv2.dilate(greenMask, kernel, iterations=1)

    redMask = cv2.inRange(hsv, redLower, redUpper)
    redMask = cv2.erode(redMask, kernel, iterations=2)
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
    redMask = cv2.dilate(redMask, kernel, iterations=1)

    center = None
    # szukanie konturów
    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>0:
        colorIndex = 0
        bpoints[bindex].appendleft(center)

    (cnts, _) = cv2.findContours(greenMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        colorIndex = 1
        gpoints[gindex].appendleft(center)

    (cnts, _) = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        colorIndex = 2
        rpoints[rindex].appendleft(center)


    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear All
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]

                bindex = 0
                gindex = 0
                rindex = 0

                paintWindow[67:,:,:] = 255

    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1

    points = [bpoints, gpoints, rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
