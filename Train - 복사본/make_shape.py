# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:11:15 2025

@author: pc
"""

# 필요한 라이브러리 임포트
import os
import cv2
import numpy as np
import random
import math

# --- 설정값 정의 ---
IMAGE_SIZE = (224, 224)
NUM_IMAGES_PER_CLASS = 1000
OUTPUT_DIR = 'dataset'

# --- 데이터셋 폴더 생성 ---
def create_directories():
    """'dataset/circle', 'dataset/oval', 'dataset/hexagon' 폴더를 생성합니다."""
    base_path = OUTPUT_DIR
    circle_path = os.path.join(base_path, 'circle')
    oval_path = os.path.join(base_path, 'oval')
    hexagon_path = os.path.join(base_path, 'hexagon') # 육각형 폴더 추가
    
    os.makedirs(circle_path, exist_ok=True)
    os.makedirs(oval_path, exist_ok=True)
    os.makedirs(hexagon_path, exist_ok=True)
    
    return circle_path, oval_path, hexagon_path

# --- 이미지 생성 함수 ---
def generate_images(circle_path, oval_path, hexagon_path):
    """원형, 타원형, 육각형의 '각진 다각형' 이미지를 생성하여 저장합니다."""
    
    print(f"'{OUTPUT_DIR}' 폴더에 이미지 생성을 시작합니다...")
    img_w, img_h = IMAGE_SIZE
    center_x, center_y = img_w // 2, img_h // 2

    # 1. 원형(에 가까운 다각형) 이미지 생성
    for i in range(NUM_IMAGES_PER_CLASS):
        image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        points = []
        num_vertices = 36 # 36개의 꼭짓점을 가진 다각형으로 원을 근사
        # ★★★ 변경된 부분: 화면에 꽉 차도록 반지름 증가 ★★★
        radius = random.uniform(img_w * 0.48, img_w * 0.5)
        
        for j in range(num_vertices):
            angle = 2 * math.pi * j / num_vertices
            # ★★★ 변경된 부분: 왜곡(noise) 정도 감소 ★★★
            noise = random.uniform(-2, 2)
            x = int(center_x + (radius + noise) * math.cos(angle))
            y = int(center_y + (radius + noise) * math.sin(angle))
            points.append([x, y])
            
        cv2.fillPoly(image, [np.array(points)], (255))
        cv2.imwrite(os.path.join(circle_path, f'circle_{i+1}.png'), image)

    print(f"- 원형 이미지 {NUM_IMAGES_PER_CLASS}개 생성 완료.")

    # 2. 타원형(에 가까운 다각형) 이미지 생성
    for i in range(NUM_IMAGES_PER_CLASS):
        image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        points = []
        num_vertices = 36
        # ★★★ 변경된 부분: 긴 쪽이 화면에 닿도록 장축 크기 증가 ★★★
        major_axis = random.uniform(img_w * 0.48, img_w * 0.5)
        minor_axis = random.uniform(major_axis * 0.6, major_axis * 0.85)
        angle_rad = math.radians(random.randint(0, 180))

        for j in range(num_vertices):
            theta = 2 * math.pi * j / num_vertices
            # ★★★ 변경된 부분: 왜곡(noise) 정도 감소 ★★★
            noise = random.uniform(-2, 2)
            # 타원 방정식과 회전 변환을 이용하여 꼭짓점 계산
            x_unit = (major_axis + noise) * math.cos(theta)
            y_unit = (minor_axis + noise) * math.sin(theta)
            x = int(center_x + x_unit * math.cos(angle_rad) - y_unit * math.sin(angle_rad))
            y = int(center_y + x_unit * math.sin(angle_rad) + y_unit * math.cos(angle_rad))
            points.append([x, y])

        cv2.fillPoly(image, [np.array(points)], (255))
        cv2.imwrite(os.path.join(oval_path, f'oval_{i+1}.png'), image)

    print(f"- 타원형 이미지 {NUM_IMAGES_PER_CLASS}개 생성 완료.")

    # 3. 육각형 이미지 생성
    for i in range(NUM_IMAGES_PER_CLASS):
        image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        points = []
        num_vertices = 6
        # ★★★ 변경된 부분: 화면에 꽉 차도록 반지름 증가 ★★★
        radius = random.uniform(img_w * 0.48, img_w * 0.5)
        
        for j in range(num_vertices):
            angle = 2 * math.pi * j / num_vertices + math.radians(30) # 육각형이 바로 서도록 30도 회전
            # ★★★ 변경된 부분: 왜곡(noise) 정도 감소 ★★★
            noise_x = random.randint(-2, 2)
            noise_y = random.randint(-2, 2)
            x = int(center_x + radius * math.cos(angle) + noise_x)
            y = int(center_y + radius * math.sin(angle) + noise_y)
            points.append([x, y])
            
        cv2.fillPoly(image, [np.array(points)], (255))
        cv2.imwrite(os.path.join(hexagon_path, f'hexagon_{i+1}.png'), image)

    print(f"- 육각형 이미지 {NUM_IMAGES_PER_CLASS}개 생성 완료.")


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    circle_dir, oval_dir, hexagon_dir = create_directories()
    generate_images(circle_dir, oval_dir, hexagon_dir)
    
    print("\n데이터셋 생성이 완료되었습니다.")
    print(f"이제 'train_shape_model.py'를 실행하여 모델을 다시 학습시켜주세요.")

