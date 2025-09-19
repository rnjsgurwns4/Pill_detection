
import csv
import os

# 문자열 유사도(레벤슈타인 거리) 계산 함수
def calculate_string_similarity(s1, s2):
    """
    두 문자열 간의 유사도를 0과 1 사이의 값으로 계산합니다. (1에 가까울수록 유사)
    레벤슈타인 거리를 정규화하여 사용합니다.
    """
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 1.0 if len1 == len2 else 0.0

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1): dp[i][0] = i
    for j in range(len2 + 1): dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution
    
    distance = dp[len1][len2]
    similarity = 1.0 - (distance / max(len1, len2))
    return similarity

def load_database(db_path):
    """
    CSV 파일을 읽어 알약 데이터베이스를 리스트 형태로 불러오기
    """
    if not os.path.exists(db_path):
        return []
        
    db = []
    try:
        # 한글 CSV 파일을 읽기 위해 인코딩 방식 지정
        with open(db_path, newline='', encoding='cp949') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                db.append(row)
        print(f"'{db_path}' 데이터베이스를 성공적으로 불러왔습니다.")
        return db
    except Exception as e:
        print(f"데이터베이스 로딩 중 오류 발생: {e}")
        return []

def find_best_match1(database, shape_candidates, color_candidates):
    """
    분석된 모양/색상 후보군과 DB를 비교하여 가장 가능성 높은 알약을 찾기
    """
    found_pills = []

    for pill_info in database:
        # DB의 모양/색상 정보가 분석된 후보 목록에 있는지 확인
        shape_match = pill_info['shape'] in shape_candidates
        color_match = pill_info['color'] in color_candidates

        if shape_match and color_match:
            # 신뢰도 점수 계산 (후보 목록의 순위가 높을수록(인덱스가 낮을수록) 점수가 낮음)
            score = (shape_candidates.index(pill_info['shape']) +
                     color_candidates.index(pill_info['color']))
            found_pills.append({'pill_info': pill_info.get('name', '알 수 없음'), 'score': score})
            

    if not found_pills:
        return '데이터베이스에서 일치하는 알약을 찾을 수 없습니다.'

    # 점수가 가장 낮은 (가장 신뢰도 높은) 알약을 정렬하여 이름 반환
    return sorted(found_pills, key=lambda x: x['score'])


def find_best_match(db, shape_candidates, color_candidates, imprint_text):
    """
    모양, 색상, 각인 정보를 종합하여 알약을 찾되,
    각인 일치 후보가 없으면 모양/색상만으로 다시 검색
    """
    imprint_text = imprint_text.upper().strip()

    # --- 1차 시도: 모양 + 색상 + 각인 모두 고려하여 검색 ---
    found_pills_full_match = []
    if imprint_text: # OCR이 각인을 찾았을 경우에만 1차 시도
        for pill in db:
            try:
                db_shape = pill.get('shape', '').strip()
                db_color = pill.get('color', '').strip()
                db_text1 = pill.get('text', '').strip().upper()
                db_text2 = pill.get('text2', '').strip().upper()

                if not (db_shape in shape_candidates and db_color in color_candidates):
                    continue

                # 문자열 유사도 계산
                sim1 = calculate_string_similarity(imprint_text, db_text1) if db_text1 else 0
                sim2 = calculate_string_similarity(imprint_text, db_text2) if db_text2 else 0
                best_similarity = max(sim1, sim2)

                # 유사도가 70% 이상일 때만 유효한 후보로 간주
                if best_similarity >= 0.7:
                    shape_score = shape_candidates.index(db_shape)
                    color_score = color_candidates.index(db_color)
                    imprint_score = (1.0 - best_similarity) * 10
                    total_score = shape_score + color_score + imprint_score
                    found_pills_full_match.append({'pill_info': pill.get('name', '알 수 없음'), 'score': round(total_score, 2)})
            except (ValueError, KeyError):
                continue

    # 1차 시도에서 후보를 찾았다면, 바로 결과를 반환
    if found_pills_full_match:
        print("    - [1차 검색 성공] 모양+색상+각인 정보로 후보를 찾았습니다.")
        return sorted(found_pills_full_match, key=lambda x: x['score'])

    # --- 2차 시도: 1차 시도 실패 시, 모양 + 색상만으로 검색 ---
    if imprint_text:
        print("    - [1차 검색 실패] 각인 일치 후보 없음. 모양+색상만으로 2차 검색을 시도합니다.")
    
    found_pills_partial_match = []
    for pill in db:
        try:
            db_shape = pill.get('shape', '').strip()
            db_color = pill.get('color', '').strip()

            if db_shape in shape_candidates and db_color in color_candidates:
                shape_score = shape_candidates.index(db_shape)
                color_score = color_candidates.index(db_color)
                # 2차 검색 결과에는 페널티 점수 20점을 부여하여 구분
                total_score = shape_score + color_score + 20
                found_pills_partial_match.append({'pill_info': pill.get('name', '알 수 없음'), 'score': round(total_score, 2)})
        except (ValueError, KeyError):
            continue
            
    if not found_pills_partial_match:
        return []

    return sorted(found_pills_partial_match, key=lambda x: x['score'])