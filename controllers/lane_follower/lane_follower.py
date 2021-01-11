import cv2
import numpy as np
from vehicle import Driver


CONTROL_COEFFICIENT = 0.002
SHOW_IMAGE_WINDOW = False


def on_cv_image_click_event(event, x, y, _, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('You clicked on a pixel with color:', img[y, x])


def regulate(tesla, camera):
    # Take the image from the camera and cut only the bottom part
    img = camera.getImage()
    img = np.frombuffer(img, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = img[380:420, :]

    # Segment the image by color in HSV color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img, np.array([70, 120, 170]), np.array([120, 160, 210]))

    # Find the largest segmented contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_center = cv2.moments(largest_contour)
    center_x = int(largest_contour_center['m10'] / largest_contour_center['m00'])

    # Find error (the lance distance from target)
    error = center_x - 120
    tesla.setSteeringAngle(error * CONTROL_COEFFICIENT)

    if SHOW_IMAGE_WINDOW:
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', on_cv_image_click_event, img)
        cv2.waitKey(0)


def main():
    tesla = Driver()
    timestep = int(tesla.getBasicTimeStep())
    
    # Init camera
    camera = tesla.getDevice('camera')
    camera.enable(timestep)
    camera.recognitionEnable(timestep)
    camera.enableRecognitionSegmentation()

    tesla.setCruisingSpeed(10)
    step_divider = 0
    while tesla.step() != -1:
        if step_divider % 2 == 0:
            regulate(tesla, camera)
        step_divider += 1


if __name__ == "__main__":
    main()
