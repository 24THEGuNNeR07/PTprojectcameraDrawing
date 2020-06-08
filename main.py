import numpy as np
import cv2
from collections import deque
import keyboard

# warunki brzegowe dla koloru niebieskiego
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# warunki brzegowe dla koloru czerwonego
redLower = np.array([0, 50, 20])
redUpper = np.array([5, 255, 255])

# warunki brzegowe dla koloru zielonego
greenLower = np.array([40, 40, 40])
greenUpper = np.array([70, 255, 255])

kernel = np.ones((5, 5), np.uint8)

# punkty rysowane

bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

paintWindow = np.zeros((471, 636, 3)) + 255

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

camera = cv2.VideoCapture(0)

# pętla główna

while True:
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not grabbed:
        break

    # szukanie niebieskich konturów

    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        colorIndex = 0
    else:
        # szukanie zielonych konturów

        greenMask = cv2.inRange(hsv, greenLower, greenUpper)
        greenMask = cv2.erode(greenMask, kernel, iterations=2)
        greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_OPEN, kernel)
        greenMask = cv2.dilate(greenMask, kernel, iterations=1)

        (cnts, _) = cv2.findContours(greenMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            colorIndex = 1
        else:

            # szukanie czerwonych konturów

            redMask = cv2.inRange(hsv, redLower, redUpper)
            redMask = cv2.erode(redMask, kernel, iterations=2)
            redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
            redMask = cv2.dilate(redMask, kernel, iterations=1)

            (cnts, _) = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts) > 0:
                colorIndex = 2

    center = None

    # jeśli kontur znaleziony

    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # koło w okół kontury
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # rysowanie (indeks -> kolor)

        if colorIndex == 0:
            bpoints[bindex].appendleft(center)
        elif colorIndex == 1:
            gpoints[gindex].appendleft(center)
        elif colorIndex == 2:
            rpoints[rindex].appendleft(center)
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

    # wyjście z programu
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # zapis obrazu
    if keyboard.is_pressed('s'):
        cv2.imwrite('image.png', paintWindow)
    # wyszyszczenie obrazu
    if keyboard.is_pressed('c'):
        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]

        bindex = 0
        gindex = 0
        rindex = 0

        paintWindow[67:, :, :] = 255

camera.release()
cv2.destroyAllWindows()
