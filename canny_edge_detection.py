import cv2 as cv

from utilities import max_screen

# This is the index of the used camera it can be 0, 1, ..
cam_index = 0
def detecting_canny_edge():
    video_capture = cv.VideoCapture(cam_index)
    window_name = "AILabo"
    max_screen(window_name, video_capture)
    while True:
        ret, frame = video_capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blur, 10, 70)
        ret, mask = cv.threshold(canny, 127, 255, cv.THRESH_BINARY)
        cv.imshow(window_name, mask)
        if cv.waitKey(1) == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()
