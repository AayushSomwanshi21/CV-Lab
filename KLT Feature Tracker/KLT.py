import cv2
import numpy as np

cap = cv2.VideoCapture('sample.webm')

ret, frame = cap.read()

old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Good Features using Shi-Tomasi function
p0 = cv2.goodFeaturesToTrack(
    old_frame, maxCorners=100, qualityLevel=0.3, minDistance=7, mask=None, blockSize=7)

# Create zero array of same dimension and data type
mask = np.zeros_like(frame)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # epsilion = 0.03 and max_iterations = 10
    # Returns new corners , status of whether they were tracked correctly and error
    p1, status,  err = cv2.calcOpticalFlowPyrLK(old_frame, curr_frame, p0, None, winSize=(
        15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # New and Old Good features
    good_new: list = p1[status == 1]
    good_old: list = p0[status == 1]

    # Draw the lines fromm old frame to the new frame(current frame)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = old.ravel()
        c, d = new.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)

        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

    # print("Frame shape:", frame.shape)
    # print("Mask shape:", mask.shape)
    # combine the tracked lines and the frame

    img = cv2.add(frame, mask)
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) == ord('q'):
        break

    # Make the current frame as the old frame and new tracked points p1 as the old tracked features
    old_frame = curr_frame.copy()
    p0 = good_new.reshape(-1, 1, 2)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
