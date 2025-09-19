

import cv2
import numpy as np

# --- 배경 제거 함수 (HSV 자동 범위 감지) ---
def remove_background_HSV_8(pill_image):
    """
    HSV 색상 공간을 이용하여 빛 반사에 강인하게 알약의 배경을 제거
    """
    if pill_image is None or pill_image.size == 0:
        return np.zeros((100, 100, 3), dtype="uint8")

    hsv = cv2.cvtColor(pill_image, cv2.COLOR_BGR2HSV)
    h, w, _ = pill_image.shape

    center_x, center_y = w // 2, h // 2
    patch_size = min(w, h) // 4
    center_patch = hsv[
        center_y - patch_size // 2 : center_y + patch_size // 2,
        center_x - patch_size // 2 : center_x + patch_size // 2
    ]

    if center_patch.size > 0:
        median_hue = np.median(center_patch[:, :, 0])
        hue_tolerance = 18
        lower_h = max(0, median_hue - hue_tolerance)
        upper_h = min(179, median_hue + hue_tolerance)
        
        # ★★★ 변경된 부분: 데이터 타입을 uint8로 명시하여 오류 수정 ★★★
        lower_bound = np.array([lower_h, 40, 50], dtype=np.uint8)
        upper_bound = np.array([upper_h, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        if median_hue < hue_tolerance:
            lower_bound2 = np.array([179 - (hue_tolerance - median_hue), 40, 50], dtype=np.uint8)
            upper_bound2 = np.array([179, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
            mask = cv2.bitwise_or(mask, mask2)
    else:
        gray = cv2.cvtColor(pill_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return pill_image

    pill_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros(pill_image.shape[:2], dtype="uint8")
    cv2.drawContours(final_mask, [pill_contour], -1, 255, -1)
    result = cv2.bitwise_and(pill_image, pill_image, mask=final_mask)
    
    return result, final_mask


# --- GrabCut 알고리즘 기반 배경 제거 ---
def remove_background(pill_image):
    """
    GrabCut 알고리즘을 사용하여 내부 질감이 복잡하거나 배경과 색이 비슷한
    이미지에서도 알약을 정교하게 분리
    """
    if pill_image is None or pill_image.size == 0:
        blank_mask = np.zeros((100, 100), dtype="uint8")
        blank_image = np.zeros((100, 100, 3), dtype="uint8")
        return blank_image, blank_mask

    mask = np.zeros(pill_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    h, w = pill_image.shape[:2]
    if h < 20 or w < 20:
        gray = cv2.cvtColor(pill_image, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(pill_image, pill_image, mask=final_mask)
        return result, final_mask

    # 알약이 이미지에 꽉 차 있을 것을 대비해, 마진을 1픽셀로 최소화
    rect = (1, 1, w - 2, h - 2)

    try:
        cv2.grabCut(pill_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"    - GrabCut 실패: {e}. Otsu 방식으로 전환합니다.")
        gray = cv2.cvtColor(pill_image, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(pill_image, pill_image, mask=final_mask)
        return result, final_mask

    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    result = pill_image * final_mask[:, :, np.newaxis]
    
    final_mask_255 = final_mask * 255
    
    return result, final_mask_255


# --- ★ 수정된 부분: 그레이스케일 기반 각인 전처리 함수에 노이즈 제거 추가 ---
def preprocess_for_imprint(original_pill_image, pill_mask):
    """
    원본 알약 이미지와 마스크를 사용하여 각인을 강조하는 전처리를 수행합니다.
    """
    gray = cv2.cvtColor(original_pill_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(equalized, (3, 3), 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    """
    # 이진화(Thresholding)를 통해 희미한 글자를 선명한 흰색으로 만듭니다.
    _, thresholded = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ★★★ 추가된 부분: 형태학적 열림(Opening) 연산으로 노이즈 제거 ★★★
    # 작은 커널을 사용하여 이미지에서 작은 점 같은 노이즈를 제거합니다.
    opening_kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    """
    
    # 최종적으로 마스크를 적용하여 배경(알약이 아닌 부분)을 제거
    imprint_only = cv2.bitwise_and(blackhat, blackhat, mask=pill_mask)
    
    return imprint_only



