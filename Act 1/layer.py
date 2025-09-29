import cv2
import numpy as np

img = cv2.imread("./63.jpg")

b, g, r = cv2.split(img)

zeros = np.zeros_like(b)

red_img = cv2.merge([zeros, zeros, r])
green_img = cv2.merge([zeros, g, zeros])
blue_img = cv2.merge([b, zeros, zeros])

cv2.imshow("Red Channel", red_img)
cv2.imshow("Green Channel", green_img)
cv2.imshow("Blue Channel", blue_img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
