import cv2

cap = cv2.VideoCapture(
    'C:/Users/Lenovo/Documents/VSCODE/CV Lab/house exploding meme - YouTube - Google Chrome 2024-03-11 19-32-42.mp4')

while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
