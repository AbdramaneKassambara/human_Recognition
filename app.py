import cv2
cap = cv2.VideoCapture("test2.mp4")
human_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

while True:
    ret, frame = cap.read()
    if frame is None:
        print("La vidéo est terminée")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.1, 9, 1)

    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
