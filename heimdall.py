import sys
import os
import time
import argparse
import numpy
import cv2
import movidius
import imutils
from imutils.video import VideoStream

arguments = argparse.ArgumentParser(
    description="Heimdall AI Camera PoC using Intel® Movidius™ Neural Compute Stick.")

arguments.add_argument('-s', '--source', type=int,
                       default=0,
                       help="Index of the video device. ex. 0 for /dev/video0")
arguments.add_argument('-pi', '--pi_cam', type=bool,
                       default=False,
                       help="Enable raspberry pi cam support. Only use this if your sure you need it.")
arguments.add_argument('-om', '--object_match_threshold', type=float,
                       default=0.9,
                       help="Percentage required for a mobile net object detection match in range of (0.0 - 1.0).")
arguments.add_argument('-mn', '--mobile_net', type=str,
                       default='ssd_mobilenet_ncs.graph',
                       help="Path to the mobile net neural network graph file.")
arguments.add_argument('-ml', '--mobile_net_labels', type=str,
                       default='labels.txt',
                       help="Path to labels file for mobile net.")
arguments.add_argument('-fps', '--show_fps', type=bool,
                       default=False,
                       help="Show fps your getting after processing.")
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

# src is the source image to convert


def bgr2rgb(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# src is a cv2.imread to resize


def resize(src, size=(320, 240)):
    return cv2.resize(src, size)


# src is a cv2.imread you want to crop
# x is the starting x position
# y is the starting y position
# width is the width of the area you want
# height is the height of the area you want


def crop(src, x, y, width, height):
    return src[y:y+height, x:x+width]


# src is the image to whiten


def whiten(src):
    image = src.copy()
    mean = numpy.mean(image)
    deviation = numpy.std(image)
    adjusted = numpy.maximum(deviation, 1.0 / numpy.sqrt(image.size))

    return numpy.multiply(numpy.subtract(image, mean), 1 / adjusted)

# src is the image to darken


def darken(src):
    image = src.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# src is the image to scale


def subscale(src):
    image = cv2.resize(src, tuple([300, 300])).astype(numpy.float16)
    return (image - numpy.float16([127.5, 127.5, 127.5])) * 0.00789


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
    mobilenet = movidius.allocate("mobilenet", ARGS.mobile_net, device)
    video = VideoStream(src=ARGS.source, usePiCamera=ARGS.pi_cam,
                        resolution=(640, 480), framerate=30).start()

    time.sleep(2.0)
    run_time = time.time()

    frames = 0
    while True:
        frame = video.read()
        if (frame is None):
            print("[ERROR] can't get frame data from camera?")
            break

        detections = movidius.ssd(inferance_objects(
            frame, mobilenet), ARGS.object_match_threshold, frame.shape)
        detections_range = range(0, detections['num_detections'])
        classifications = {}

        for i in detections_range:
            classifications[i] = (detections.get(
                'detection_classes_' + str(i)), None, (255, 255, 0))

        frames += 1
        if (time.time() - run_time) > 1:
            if (ARGS.show_fps):
                print("FPS: " + str(frames / (time.time() - run_time)))

            frames = 0
            run_time = time.time()

        overlay_detections(frame, detections, classifications)
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    movidius.deattach(0)
    print("Deattached from movidius device.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
