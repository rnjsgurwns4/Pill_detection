
import cv2
import numpy as np
from ultralytics import YOLO


#runs_new/detect/train/weights/best.pt
def detect_pills1(image_path, model_path='runs_new/detect/train/weights/best.pt'):
    """
    YOLOv8 모델을 사용하여 이미지에서 알약 객체를 탐지
    """
    try:
        # force_reload=True 옵션은 캐시 문제 방지에 도움이 될 수 있습니다.
        model_path = 'best.pt'
        model = YOLO(model_path)
        results = model.predict(source=image_path, save=False, conf=0.3, iou=0.25)
        boxes = []
        # results.xyxy[0]는 탐지된 객체 정보를 담고 있습니다.
        if len(results) > 0:
            for box in results[0].boxes.xyxy:
                # box 텐서에서 좌표 값을 추출합니다.
                x1, y1, x2, y2 = map(int, box)
                boxes.append([x1, y1, x2, y2])
        print(f"총 {len(boxes)}개의 알약이 탐지되었습니다.")
        return boxes
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        print("YOLOv8 모델 로딩에 실패했습니다. 'best.pt' 파일 경로를 확인하세요.")
        print("임시로 더미 바운딩 박스를 반환합니다.")
        dummy_image = cv2.imread(image_path)
        if dummy_image is None:
            # 이미지가 없을 경우를 대비해 600x800 크기의 흰색 이미지를 생성합니다.
            dummy_image = np.full((600, 800, 3), 255, dtype=np.uint8)
        h, w, _ = dummy_image.shape
        return [
            [int(w*0.1), int(h*0.1), int(w*0.4), int(h*0.4)],
            [int(w*0.6), int(h*0.6), int(w*0.9), int(h*0.9)]
        ]
    
def detect_pills(image_path, model_path='best.pt'):
    """
    YOLOv8 모델을 사용하여 이미지에서 알약 객체를 탐지하고,
    수동 NMS(비최대 억제)를 적용하여 중복 박스를 확실하게 제거합니다.
    """
    try:
        model_path = 'best.pt'
        model = YOLO(model_path)
        
        # 1. YOLO 예측 실행 (iou 파라미터는 제거하고 원시 결과 받기)
        results = model.predict(source=image_path, save=False, conf=0.3)
        
        raw_boxes = []
        confidences = []

        if len(results) > 0:
            for box in results[0].boxes:
                # bounding box 좌표와 신뢰도(confidence)를 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # OpenCV NMS 함수 형식에 맞게 (x, y, w, h) 형태로 저장
                raw_boxes.append([x1, y1, x2 - x1, y2 - y1]) 
                confidences.append(conf)
                
        if not raw_boxes:
            print("탐지된 알약이 없습니다.")
            return []

        # 2. OpenCV의 NMSBoxes 함수를 사용하여 중복 박스 제거
        #    score_threshold: 이 신뢰도 이하의 박스는 고려하지 않음
        #    nms_threshold: 이 iou 값 이상 겹치는 박스는 중복으로 보고 제거
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, score_threshold=0.25, nms_threshold=0.4)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = raw_boxes[i]
                final_boxes.append([x, y, x + w, y + h])
                
        print(f"총 {len(final_boxes)}개의 알약이 탐지되었습니다. (중복 제거 완료)")
        return final_boxes
        
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        print("YOLOv8 모델 로딩에 실패했습니다. 'best.pt' 파일 경로를 확인하세요.")
        print("임시로 더미 바운딩 박스를 반환합니다.")
        dummy_image = cv2.imread(image_path)
        if dummy_image is None:
            dummy_image = np.full((600, 800, 3), 255, dtype=np.uint8)
        h, w, _ = dummy_image.shape
        return [
            [int(w*0.1), int(h*0.1), int(w*0.4), int(h*0.4)],
        ]