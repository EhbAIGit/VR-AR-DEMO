import cv2 as cv

from utilities import max_screen
# This is the index of the used camera it can be 0, 1, ..
cam_index = 0
def openpose():
    net = cv.dnn.readNetFromTensorflow("graph_opt.pb") #weights

    in_width = 368
    in_height = 368
    thr = 0.2
    window_name = "AILabo"
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

    video_capture = cv.VideoCapture(cam_index)
    max_screen(window_name, video_capture)
    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        # Break the loop when the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (in_width, in_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]
        assert (len(BODY_PARTS) == out.shape[1])
        points = [] # list

        for i in range(len(BODY_PARTS)):
            heat_map = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heat_map)
            x = (frame_width * point[0]) / out.shape[3]
            y = (frame_height * point[1]) / out.shape[2]
            #Add a point if its confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > thr else None)

        for pair in POSE_PAIRS:
            part_from = pair[0]
            part_to = pair[1]
            assert (part_from in BODY_PARTS)
            assert (part_to in BODY_PARTS)

            id_from = BODY_PARTS[part_from]
            id_to = BODY_PARTS[part_to]

            if points[id_from] and points[id_to]:
                cv.line(frame, points[id_from], points[id_to], (0,255,0),3)
                cv.ellipse(frame, points[id_from], (3,3), 0,0,360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[id_to], (3,3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency () / 1000
        cv.putText(frame, '%.2fms' % (t/freq), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv.imshow(window_name, frame)

    # Release the webcam and close the window
    video_capture.release()
    cv.destroyAllWindows()