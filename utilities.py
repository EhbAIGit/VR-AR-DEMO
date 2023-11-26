import cv2 as cv
def max_screen(window_name, video_capture):
    if not video_capture.isOpened():
        print("Can't Open Camera")
        exit()
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
