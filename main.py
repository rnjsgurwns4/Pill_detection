

import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image 

# 로컬 모듈 임포트
from object_detection import detect_pills
from image_preprocessing import remove_background, preprocess_for_imprint
from color_analysis import get_dominant_color
from shape_analysis import classify_shape_with_ai
from database_handler import load_database, find_best_match
from imprint_analysis import get_imprint 
#from imprint_analysis_naver import get_imprint

# 한글 텍스트를 이미지에 그리는 함수
def draw_korean_text(image, text, position, font_path, font_size, font_color):
    """
    OpenCV 이미지 위에 한글 텍스트를 그립니다.
    """
    # OpenCV 이미지를 Pillow 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 지정된 경로의 폰트 로드
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"오류: '{font_path}' 폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    # 텍스트 그리기
    draw.text(position, text, font=font, fill=font_color)
    
    # Pillow 이미지를 다시 OpenCV 이미지로 변환
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



# --- 메인 실행 로직 ---
if __name__ == "__main__":
    IMAGE_PATH = "test_image/sample.png"
    YOLO_MODEL_PATH = 'runs_new/detect/train/weights/best.pt'
    SHAPE_MODEL_PATH = "shape_model.h5"
    OUTPUT_DIR = "output_images"
    DB_PATH = "pill.csv"
    FONT_PATH = "malgun.ttf" 

    # 결과 이미지를 저장할 폴더 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        print(f"오류: '{IMAGE_PATH}' 이미지를 찾을 수 없습니다.")
        exit()
        
    # AI 모델을 루프 시작 전에 딱 한 번만 로드
    shape_model = None
    try:
        shape_model = load_model(SHAPE_MODEL_PATH)
        print(f"'{SHAPE_MODEL_PATH}' 모양 분류 모델을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: '{SHAPE_MODEL_PATH}' 모델을 불러올 수 없습니다. {e}")
        print("모양 분석 기능이 비활성화됩니다.")
    
    # 데이터베이스 로드
    pill_db = load_database(DB_PATH)
    if not pill_db:
        print(f"오류: '{DB_PATH}' 데이터베이스를 불러올 수 없습니다.")
        exit()

    # 1. YOLOv8로 알약 탐지
    pill_boxes = detect_pills(IMAGE_PATH, YOLO_MODEL_PATH)
    
    for i, box in enumerate(pill_boxes):
        x1, y1, x2, y2 = box
        # 원본 이미지에서 알약 부분만 잘라내기
        cropped_pill = original_image[y1:y2, x1:x2]
        output_path = os.path.join(OUTPUT_DIR, f"pill_{i+1}.jpg")
        cv2.imwrite(output_path, cropped_pill)
        
        print(f"\n--- 알약 #{i+1} 분석 시작 ---")
        
        # 2. 배경 제거
        pill_without_bg, pill_mask = remove_background(cropped_pill.copy())

        # 3. 색상 분석
        dominant_color_rgb, color_candidates = get_dominant_color(pill_without_bg)
        print(f"  - 주요 색상 후보: {color_candidates} (대표 RGB: {dominant_color_rgb})")
        
        # 4. 모양 분석을 위한 이진화
        gray_pill = cv2.cvtColor(pill_without_bg, cv2.COLOR_BGR2GRAY)
        _, binarized_image = cv2.threshold(gray_pill, 10, 255, cv2.THRESH_BINARY)
        
        # 컨투어 근사(Approximation)를 이용한 스마트 스무딩
        smoothed_binarized_image = binarized_image.copy()
        contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            pill_contour = max(contours, key=cv2.contourArea)
            
            # 컨투어를 더 부드러운 다각형으로 근사합니다.
            perimeter = cv2.arcLength(pill_contour, True)
            # epsilon 값을 작게 설정하여(0.005) 미세한 스무딩 효과를 줍니다.
            epsilon = 0.005 * perimeter 
            approximated_contour = cv2.approxPolyDP(pill_contour, epsilon, True)
            
            # 새로운 검은색 이미지에 근사된 컨투어를 그려넣어 깨끗한 모양을 만듭니다.
            smoothed_binarized_image = np.zeros_like(binarized_image)
            cv2.drawContours(smoothed_binarized_image, [approximated_contour], -1, (255), -1)
        
        
        # 5. AI로 모양 분석
        shape_result = "모델 로드 실패"
        if shape_model: # 모델이 성공적으로 로드된 경우에만 분석 실행
            shape_result = classify_shape_with_ai(smoothed_binarized_image, shape_model)
        
        print(f"  - AI 모양 분석 결과: {shape_result}")
        
        # 6. 각인(글자) 강조를 위한 전처리
        imprint_image = preprocess_for_imprint(cropped_pill.copy(), pill_mask)
        
        #------각인------
        imprint_text = get_imprint(cropped_pill.copy(), pill_mask)
        print(f"  - 인식된 각인: '{imprint_text}'")
        
        # 각인 정보까지 포함하여 최종 알약 추측
        candidate_pills = find_best_match(pill_db, shape_result, color_candidates, imprint_text)
        #-----------------
        
        # 7. 결과 이미지 저장
        output_path_binarized = os.path.join(OUTPUT_DIR, f"shape_binarized_{i+1}.jpg")
        cv2.imwrite(output_path_binarized, smoothed_binarized_image)

        output_path_imprint = os.path.join(OUTPUT_DIR, f"imprint_preprocessed_{i+1}.jpg")
        cv2.imwrite(output_path_imprint, imprint_image)
        print(f"  - 전처리 이미지 저장 완료: '{output_path_binarized}', '{output_path_imprint}'")
        
        # 8. 알약 후보군 출력
        #candidate_pills = find_best_match(pill_db, shape_result, color_candidates)
        print("  ---------------------------------")
        if candidate_pills:
            print("  => 최종 식별 후보:")
            for candidate in candidate_pills:
                # 점수가 낮을수록 정확도가 높다는 것을 의미
                print(f"     - {candidate['pill_info']} (점수: {candidate['score']})")
            # 가장 신뢰도 높은 후보(점수가 가장 낮은 후보)를 가져옴
            top_candidate = candidate_pills[0]
            label = f"{top_candidate['pill_info']}"
            
            # 네모 박스 그리기
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 텍스트 배경 그리기
            cv2.rectangle(original_image, (x1, y1 - 25), (x1 + 200, y1), (0, 255, 0), -1)
            # Pillow를 이용해 한글 텍스트 그리기
            original_image = draw_korean_text(original_image, label, (x1, y1 - 25), FONT_PATH, 18, (0, 0, 0))
        else:
            print("  => 최종 식별 결과: 데이터베이스에서 일치하는 알약을 찾을 수 없습니다.")
        print("  ---------------------------------")
        
    output_path = os.path.join(OUTPUT_DIR, "final_result.jpg")
    cv2.imwrite(output_path, original_image)
    print("\n\n모든 분석이 완료되었습니다.")
