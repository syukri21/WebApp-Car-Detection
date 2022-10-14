import cv2


class VehicleDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)
        # Allow classes containing Vehicles only
        self.classes_allowed = [
            2,  # car
            3,  # bike
            7,  # truck
        ]

    def detect_vehicles(self, img):
        # Detect Objects
        vehicles_boxes = []
        class_ids_list = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.2:
                # Skip detection with low confidence
                continue

            if class_id in self.classes_allowed:
                class_ids_list.append(class_id)
                vehicles_boxes.append(box)

        return vehicles_boxes, class_ids_list
