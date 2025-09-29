import cv2

img = cv2.imread("./63.jpg")

resizing = cv2.resize(img, None, fy=0.1, fx=0.1)

cv2.imshow("Image", resizing)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
