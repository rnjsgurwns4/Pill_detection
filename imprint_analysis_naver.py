

import cv2
import re
import requests 
import json
import uuid
import time
import base64

# --- ★★★ 네이버 CLOVA OCR API 정보 (사용자 정보로 채워주세요) ★★★ ---
# 1. 네이버 클라우드 플랫폼에서 발급받은 API Gateway URL을 입력하세요.
API_URL = "https://s0bv93ys8h.apigw.ntruss.com/custom/v1/46232/f320cdae3a6f2f94769e5a113068691bc3df84504bee2b65d2a2a7c096f9a4b2/general" 
# 2. 네이버 클라우드 플랫폼에서 발급받은 Secret Key를 입력하세요.
SECRET_KEY = "aXlnVlpxVUtTRmtMTVV3SWRZdGRUa3dzUFJadHF1T1c="
# --------------------------------------------------------------------


# ★★★ 대폭 수정된 부분: 네이버 CLOVA OCR API를 호출하는 단일 함수 ★★★
def recognize_text_naver_ocr(image):
    """
    네이버 CLOVA OCR API를 호출하여 이미지에서 텍스트를 인식합니다.
    """
    if not API_URL or "YOUR_API_URL_HERE" in API_URL:
        print("    - 네이버 OCR API URL이 설정되지 않았습니다.")
        return ""

    # 이미지를 base64로 인코딩
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    request_json = {
        'images': [{'format': 'jpg', 'name': 'pill_image', 'data': image_base64}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    
    payload = json.dumps(request_json).encode('UTF-8')
    headers = {
      'X-OCR-SECRET': SECRET_KEY,
      'Content-Type': 'application/json'
    }

    try:
        response = requests.post(API_URL, headers=headers, data=payload)
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        result = response.json()
        
        # 인식된 텍스트들을 하나로 합침
        full_text = ""
        for field in result['images'][0]['fields']:
            full_text += field['inferText']
            
        cleaned_text = re.sub(r'[^A-Z0-9]', '', full_text.upper())
        return cleaned_text
        
    except requests.exceptions.RequestException as e:
        print(f"    - 네이버 OCR API 호출 오류: {e}")
        return ""

# ★★★ 대폭 수정된 부분: 불필요한 전처리 제거 및 로직 단순화 ★★★
def get_imprint(original_pill_image, pill_mask):
    """
    배경이 제거된 원본 이미지를 네이버 OCR로 분석하여 각인 텍스트를 추출합니다.
    """
    # 1. 마스크를 사용해 배경이 제거된 깨끗한 알약 이미지를 만듭니다.
    clean_pill_image = cv2.bitwise_and(original_pill_image, original_pill_image, mask=pill_mask)
    
    # 2. 이 깨끗한 이미지를 바로 OCR 분석에 사용합니다.
    print("    - 네이버 OCR 분석 시도...")
    imprint_text = recognize_text_naver_ocr(clean_pill_image)
    print(f"      => 결과: '{imprint_text}'")
    
    return imprint_text

