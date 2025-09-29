import cv2

image = cv2.imread("./63.jpg")

cropped = image[100:400, 150:450]

cv2.imwrite("./Wallpaper 43-cropped.jpg", cropped)

cv2.imshow("Original", image)
cv2.imshow("Cropped", cropped)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
