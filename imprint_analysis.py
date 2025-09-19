# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:40:00 2025

@author: pc
"""

import cv2
import pytesseract # ★ Naver OCR 대신 다시 Tesseract 사용
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_engraved(original_pill_image, pill_mask):
    """
    음각(홈이 파인) 각인을 강조하기 위한 전처리를 수행합니다. (Black Hat)
    """
    gray = cv2.cvtColor(original_pill_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(equalized, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    _, thresholded = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening_kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    
    cv2.imshow('',opened)
    cv2.waitKey(0)
    return cv2.bitwise_and(opened, opened, mask=pill_mask)

def preprocess_for_printed(original_pill_image, pill_mask):
    """
    양각(인쇄된) 각인을 강조하기 위한 전처리를 수행합니다. (Adaptive Thresholding)
    """
    gray = cv2.cvtColor(original_pill_image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    cv2.imshow('',thresholded)
    cv2.waitKey(0)
    return cv2.bitwise_and(thresholded, thresholded, mask=pill_mask)

def preprocess_for_dark_print_hsv(original_pill_image, pill_mask):
    """
    HSV 색상 공간의 명도(Value) 채널을 이용하여 어두운 글씨를 추출합니다.
    """
    hsv = cv2.cvtColor(original_pill_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_channel_clahe = clahe.apply(v_channel)
    _, thresholded = cv2.threshold(v_channel_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_and(thresholded, thresholded, mask=pill_mask)

# ★★★ 대폭 수정된 부분: Tesseract 최종 성능 최적화 로직 적용 ★★★
def recognize_text_with_rotation(image):
    """
    이미지를 회전, 확대, 윤곽선 강화 등 Tesseract에 최적화하여 최상의 결과를 찾습니다.
    """
    best_text = ""
    max_confidence = 0
    best_word_count = 0

    # 시도해볼 Tesseract 페이지 분할 모드(psm) 목록
    psm_modes = ['--psm 7', '--psm 8']

    for angle in range(0, 180, 15):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotated = image
        if angle > 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # 1. 이미지 확대 (Upscaling)
        upscaled = cv2.resize(rotated, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # 2. 글씨 굵게 만들기 (Dilation)
        kernel = np.ones((3, 3), np.uint8) # 커널 크기를 살짝 키워 더 확실한 효과
        dilated = cv2.dilate(upscaled, kernel, iterations=1)

        # ★ 추가된 부분: 텍스트 윤곽선 강화 (Erosion) ★
        # 굵어진 글씨를 살짝 깎아내어, 주변 노이즈를 제거하고 글자 형태를 더 명확하게 만듭니다.
        erosion_kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(dilated, erosion_kernel, iterations=1)
        
        
            
        # 3. 여러 PSM 모드로 Tesseract OCR 수행
        for psm in psm_modes:
            custom_config = f'--oem 3 {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            data = pytesseract.image_to_data(eroded, config=custom_config, output_type=pytesseract.Output.DICT)
            
            current_text, total_conf, word_count = [], 0, 0
            
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                word = data['text'][i].strip()
                if conf > 60 and word:
                    current_text.append(word)
                    total_conf += conf
                    word_count += 1
            
            if word_count > 0:
                avg_conf = total_conf / word_count
                if avg_conf > max_confidence or (abs(avg_conf - max_confidence) < 5 and word_count > best_word_count):
                    max_confidence = avg_conf
                    best_word_count = word_count
                    best_text = "".join(current_text)
                
    cleaned_text = re.sub(r'[^A-Z0-9]', '', best_text)
    return cleaned_text

def get_imprint(original_pill_image, pill_mask):
    """
    여러 전처리 방식을 모두 시도하고 Tesseract로 분석하여 최적의 결과를 선택합니다.
    """
    # 1. 분석할 후보 이미지들을 생성
    #printed_image = preprocess_for_printed(original_pill_image, pill_mask)
    engraved_image = preprocess_for_engraved(original_pill_image, pill_mask)
    #hsv_value_image = preprocess_for_dark_print_hsv(original_pill_image, pill_mask)
    
    # 2. 각 후보 이미지에 대해 OCR 실행
    print("    - (A) 인쇄 각인(Adaptive Thresh) 분석 시도...")
    text_printed = recognize_text_with_rotation(original_pill_image)
    print(f"      => 결과: '{text_printed}'")
    
    print("    - (B) 홈 파인 각인(Black Hat) 분석 시도...")
    text_engraved = recognize_text_with_rotation(engraved_image)
    print(f"      => 결과: '{text_engraved}'")
    """
    print("    - (C) 인쇄 각인(HSV Value) 분석 시도...")
    text_hsv = recognize_text_with_rotation(hsv_value_image)
    print(f"      => 결과: '{text_hsv}'")
    """

    # 3. 최상의 결과 선택 (가장 긴 텍스트를 최종 결과로 선택)
    results = [text_printed, text_engraved]
    best_result = max(results, key=len)
    
    print(f"    - 최종 선택된 각인: '{best_result}'")
    return best_result

