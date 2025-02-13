import cv2
import pytesseract
import re
from spellchecker import SpellChecker

# Tesseract 설치 경로가 기본 경로가 아닌 경우, 아래와 같이 지정합니다.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """
    이미지 전처리 함수
    - 이미지를 읽고, 그레이스케일 및 adaptive thresholding 적용
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # adaptive thresholding 적용 (노이즈 제거, 대비 강화)
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return processed

def extract_text_from_image(image):
    """
    전처리된 이미지에서 텍스트를 추출하는 함수
    """
    # 언어 설정은 필요에 따라 조정 (예: 'eng', 'kor' 등)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def validate_text(text):
    """
    추출된 텍스트에 대해 다음 검증을 수행:
    - 오타(스펠링 오류) 검사
    - 줄 간격(라인 수) 검사
    - 문장부호(예: 문장 끝에 온점) 검사
    - 불필요한 공백 검사
    """
    validation_result = {}
    errors = []

    # 1. 오타 검사 (SpellChecker는 영어 단어 기준)
    spell = SpellChecker()
    words = re.findall(r'\b\w+\b', text)
    misspelled = spell.unknown(words)
    if misspelled:
        errors.append(f"오타 발견: {', '.join(misspelled)}")
    validation_result['misspelled_words'] = list(misspelled)
    
    # 2. 줄 간격(라인 수) 검사: 예제에서는 3줄 이상이어야 한다고 가정
    lines = text.split('\n')
    # 빈 줄 제거 후 카운트
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) < 3:
        errors.append("문단이 너무 짧거나 줄 간격에 문제가 있을 수 있습니다.")
    validation_result['lines_count'] = len(non_empty_lines)
    
    # 3. 문장부호 검사: 각 문장이 온점(.)으로 끝나는지 확인 (단, 마지막 문장은 제외)
    # 간단한 예제로, 문장 구분자는 온점, 물음표, 느낌표로 가정합니다.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for i, sentence in enumerate(sentences[:-1]):  # 마지막 문장은 검증 대상에서 제외
        if not sentence.endswith('.'):
            errors.append(f"문장부호 누락: '{sentence}' (문장이 온점(.)으로 끝나야 합니다.)")
    
    # 4. 불필요한 공백 검사 (연속된 2개 이상의 공백이 있는지)
    if re.search(r' {2,}', text):
        errors.append("불필요한 여백(연속된 공백)이 있습니다.")
    
    validation_result['errors'] = errors
    validation_result['is_valid'] = len(errors) == 0
    return validation_result

def main(image_path):
    try:
        # 1. 이미지 전처리
        processed_image = preprocess_image(image_path)
        
        # 2. OCR을 통한 텍스트 추출
        text = extract_text_from_image(processed_image)
        print("----- 추출된 텍스트 -----")
        print(text)
        
        # 3. 텍스트 검증
        result = validate_text(text)
        print("\n----- 검증 결과 -----")
        print(f"오타 단어: {result['misspelled_words']}")
        print(f"유효한 줄 수: {result['lines_count']}")
        print("발견된 오류:")
        if result['errors']:
            for err in result['errors']:
                print(f"- {err}")
        else:
            print("모든 검증 항목을 통과했습니다.")
            
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    # 이미지 파일 경로를 지정하세요.
    image_path = 'sample_image.png'
    main(image_path)
