import cv2
from vehicle_detector import VehicleDetector


def get_cascade_classifiers():
    return cv2.CascadeClassifier("./cascades/car.xml")


def get_video_capture():
    return cv2.VideoCapture('./static/uploads/video.mp4')


def run_video():
    # create classifier

    # Load Veichle Detector
    vd = VehicleDetector()

    video = get_video_capture()

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter("./video/output.mp4", fourcc, 20.0, (frame_width, frame_height))

    total_fps = 0
    cut_fps = 5
    fps = 0
    while True:
        # Read  the current frame
        (read_successful, frame) = video.read()

        total_fps += 1
        fps += 1
        if fps != cut_fps:
            continue
        fps = 0

        if type(None) != type(frame):
            vehicle_boxes, _ = vd.detect_vehicles(frame)

            for box in vehicle_boxes:
                x, y, w, h = box

                cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 0, 180), 1)
                cv2.putText(frame, "Car", (x, y - 5), 1, 1, (255, 255, 0), 2)

            cv2.imshow("Cars", frame)
            cv2.waitKey(1)

        out.write(frame)

        if total_fps >= 100:
            cv2.destroyAllWindows();
            break;


if __name__ == '__main__':
    run_video()
