import sys
import os
import time
import argparse
import textwrap
import smtplib
import numpy
import cv2
import movidius
import imutils
from imutils.video import VideoStream

arguments = argparse.ArgumentParser(
    description="Heimdall AI Camera PoC using Intel Movidius Neural Compute Stick 1.")

arguments.add_argument('-s', '--source', type=int,
                       default=0,
                       help="Index of the video device. ex. 0 for /dev/video0")
arguments.add_argument('-pi', '--pi', type=bool,
                       default=False,
                       help="Enable raspberry pi cam support. Only use this if your sure you need it.")
arguments.add_argument('-m', '--match_threshold', type=float,
                       default=0.9,
                       help="Percentage required for a mobile net object detection match in range of (0.0 - 1.0).")
arguments.add_argument('-g', '--mobile_net_graph', type=str,
                       default='ssd_mobilenet_ncs.graph',
                       help="Path to the mobile net neural network graph file.")
arguments.add_argument('-l', '--mobile_net_labels', type=str,
                       default='labels.txt',
                       help="Path to labels file for mobile net.")
arguments.add_argument('-fps', '--fps', type=bool,
                       default=False,
                       help="Show fps your getting after processing.")
arguments.add_argument('-alerts', '--alerts', type=str,
                       default=None,
                       help="Classification list that triggers alerts.")
arguments.add_argument('-email', '--email', type=str,
                       default=None,
                       help="Email address to send email alerts too.")
arguments.add_argument('-email_server', '--email_server', type=str,
                       default="localhost",
                       help="Email server to send email alerts from.")
arguments.add_argument('-email_username', '--email_username', type=str,
                       default=None,
                       help="Email server username to send email alerts with.")
arguments.add_argument('-email_password', '--email_password', type=str,
                       default=None,
                       help="Email server password to send email alerts with.")
ARGS = arguments.parse_args()

# classification(s) to search for based on labels.txt
CLASS_BACKGROUND = 0
CLASS_AEROPLANE = 1
CLASS_BICYCLE = 2
CLASS_BIRD = 3
CLASS_BOAT = 4
CLASS_BOTTLE = 5
CLASS_BUS = 6
CLASS_CAR = 7
CLASS_CAT = 8
CLASS_CHAIR = 9
CLASS_COW = 10
CLASS_DINNINGTABLE = 11
CLASS_DOG = 12
CLASS_HORSE = 13
CLASS_MOTORBIKE = 14
CLASS_PERSON = 15
CLASS_POTTEDPLANT = 16
CLASS_SHEEP = 17
CLASS_SOFA = 18
CLASS_TRAIN = 19
CLASS_TV_MONITOR = 20

# classification labels


LABELS = [line.rstrip('\n')
          for line in open(ARGS.mobile_net_labels) if line != 'classes\n']

# subject for the alert
# text for the alert


def alert_smtp(subject, text):
    message = textwrap.dedent("""\
        From: %s
        To: %s
        Subject: %s
        %s
        """ % ("alerts@heimdall.py", ", ".join(ARGS.email), "Heimdall Alert: " + subject, text))

    print(message)
    server = smtplib.SMTP(ARGS.email_server)
    if (ARGS.email_username is not None and ARGS.email_password is not None):
        server.starttls()
        server.login(ARGS.email_username, ARGS.email_password)

    server.sendmail("alerts@heimdall.py", ARGS.email, message)
    server.quit()

# src is the source image to convert


def bgr2rgb(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB)


# src is a cv2.imread you want to crop
# x is the starting x position
# y is the starting y position
# width is the width of the area you want
# height is the height of the area you want


def crop(src, x, y, width, height):
    return src[y:y+height, x:x+width]


# src is the image to scale


def subscale(src):
    return (cv2.resize(src, tuple([300, 300])).astype(numpy.float16) - numpy.float16([127.5, 127.5, 127.5])) * 0.00789


# image is a cv2.imread
# color is the border color


def overlay(src, x, y, width, height, color=(255, 255, 0), thickness=2, left_label=None, right_label=None):
    cv2.rectangle(src, (x, y), (x + width, y + height), color, thickness)

    if (left_label is not None):
        cv2.putText(src, left_label, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (32, 32, 32), 2)

    if (right_label is not None):
        cv2.putText(src, right_label, (x + width - len(right_label) * 9, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (32, 32, 32), 2)

# image is a cv2.imread
# color is the border color


def overlay_detections(src, outputs, classifications):
    r = range(0, outputs['num_detections'])
    for i in r:
        classification = classifications[i]
        (y1, x1) = outputs.get('detection_boxes_' + str(i))[0]
        (y2, x2) = outputs.get('detection_boxes_' + str(i))[1]
        label = (LABELS[outputs.get('detection_classes_' + str(i))] +
                 ": " + str(outputs.get('detection_scores_' + str(i))) + "%")

        overlay(src, x1, y1, x2 - x1, y2 -
                y1, classification[2], 2, label, classification[1])


# src is a cv2.imread
# graph is the loaded facenet graph


def inferance_objects(src, graph):
    scaled = subscale(bgr2rgb(src))
    graph.LoadTensor(scaled, 'user object')
    output, t = graph.GetResult()

    return output


def main():
    device = movidius.attach(0)
    if (device is None):
        print("No movidius device was found.")
        exit()

    print("Attached to movidius device at " + device)
    mobilenet = movidius.allocate("mobilenet", ARGS.mobile_net_graph, device)
    video = VideoStream(src=ARGS.source, usePiCamera=ARGS.pi,
                        resolution=(320, 240), framerate=30).start()

    print("Waiting for camera to start...")
    time.sleep(2.0)
    run_time = time.time()
    alert_time = time.time()

    print("Running...")
    frames = 0
    while True:
        frame = video.read()
        if (frame is None):
            print("[ERROR] can't get frame data from camera?")
            break

        detections = movidius.ssd(inferance_objects(
            frame, mobilenet), ARGS.match_threshold, frame.shape)
        detections_range = range(0, detections['num_detections'])
        classifications = {}

        for i in detections_range:
            class_id = detections.get('detection_classes_' + str(i))
            label = None
            color = (255, 255, 0)

            if (ARGS.alerts is not None):
                alerts = ARGS.alerts.replace(" ", "").split(",")
                for alert in alerts:
                    if (class_id == alert):
                        color = (0, 255, 0)

                        if (time.time() - alert_time) > 120:
                            alert_smtp(class_id, class_id + " was found.")
                            alert_time = time.time()

            classifications[i] = (class_id, label, color)

        if (ARGS.fps):
            frames += 1

            if (time.time() - run_time) > 1:
                print("[FPS] " + str(frames / (time.time() - run_time)))
                frames = 0
                run_time = time.time()

        overlay_detections(frame, detections, classifications)
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    movidius.deattach(0)
    print("Deattached from movidius device at " + device)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
