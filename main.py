import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import time
import pyautogui
import pickle
import platform

import xgboost as xgb


model_path = "Model/xgb_model46.json"
label_path = "./final_y_labels_100.csv"
keyboard_path = "./keyboard_dict.p"


with open(keyboard_path, 'rb') as f:
    KEYBOARD_DICT = pickle.load(f)

FONT_SIZE = 100 # 글자 표시 사이즈
BOX_HEIGHT = 100 # 글자, 숫자 버튼 높이
COLOR = (255, 0, 0) # 글자 색깔

SPEED_LIMIT = 0.05 # 손 끝 속도 기준치
TIME_FRAME = 0.2 # 속도 계산 시간차
IS_NUM = False # 숫자 인식인지 글자 인식인지 확인하는 Boolean Flag

mp_hands = mp.solutions.hands  # hand model
mp_drawing = mp.solutions.drawing_utils  # frawing utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # detection
    image.flags.writeable = True  # image is writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results 



# OpenCV 이미지에 한글 그려주는 함수
def myPutText(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font_path = 'fonts/gulim.ttc'
    if platform.system() == "Darwin":
        font_path = 'AppleGothic.ttf'

    font = ImageFont.truetype(font_path, font_size)  # 윈도우 에서는 'fonts/gulim.ttc', 애플 "AppleGothic.ttf"
    draw.text(pos, text, font=font, fill=font_color)
    return np.array(img_pil)


# 머신러닝 detection 함수
def detect_ML(input_array):
    pred = model.predict(xgb.DMatrix(input_array.reshape(1, -1)))[0]
    pred = np.argmax(pred)
    return labels["val"][pred]

# 타이핑 함수
def keyboard(cur_res_final):
    if cur_res_final == 'ㅚ':
        pyautogui.write(KEYBOARD_DICT['ㅗ'])
        pyautogui.write(KEYBOARD_DICT['ㅣ'])
    elif cur_res_final == 'ㅟ':
        pyautogui.write(KEYBOARD_DICT['ㅜ'])
        pyautogui.write(KEYBOARD_DICT['ㅣ'])
    elif cur_res_final == 'ㅢ':
        pyautogui.write(KEYBOARD_DICT['ㅡ'])
        pyautogui.write(KEYBOARD_DICT['ㅣ'])
    else:
        pyautogui.write(KEYBOARD_DICT[cur_res_final])

cap = cv2.VideoCapture(0)
landmark_input = deque()

cur_res_final = ""
prev_res_final = ""

labels = pd.read_csv(label_path, index_col=0)
double_const = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}

final_type_store = []

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 화면 너비
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 화면 높이


model = xgb.Booster()
model.load_model(model_path)


prev_hand_end = np.full((5, 3), 0.5)  # 손가락 끝 좌표를 저장할 변수
prev_time = time.time()  # 속도계산을 위한 시간 저장 변수

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
    ) as hands:
        image, results = mediapipe_detection(frame, hands)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):   # idx = 0 : right, idx = 1 : left
                left_hand = False       # 왼손 인식 여부
                if len(results.multi_hand_landmarks) == 1: # 한 손만 인식 될 때 (오른손)
                    # 손 랜드마크 이미지에 그리기
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS           
                    )

                    # 오른손 랜드마크 np.array (3D)로 변환
                    hand_array = np.array(
                        [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
                    )

                    # 손 끝 (엄지-새끼) 랜드마크만 추출
                    cur_hand_end = hand_array[[4, 8, 12, 16, 20], :]

                    # 원본 손 랜드마크 좌표에서 모델로 인식할 때에는 x, y만 사용
                    hand_array = hand_array[:, :2].flatten()

                elif len(results.multi_hand_landmarks) == 2: # 두 손이 인식 될 때
                    # 손 랜드마크 이미지에 그리기
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS         
                    )
                    if idx == 0:  # 오른손
                        # 손 랜드마크 np.array (3D)로 변환
                        hand_array = np.array(
                            [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
                        )

                        # 손 끝 (엄지-새끼) 랜드마크만 추출
                        cur_hand_end = hand_array[[4, 8, 12, 16, 20], :]
                        # 원본 손 랜드마크 좌표에서 모델로 인식할 때에는 x, y만 사용
                        hand_array = hand_array[:, :2].flatten()
                    elif idx == 1:  # 왼손
                        left_hand = True    # 왼손 인식

                        # 손 랜드마크 np.array (3D)로 변환
                        left_hand_array = np.array(
                            [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
                        )
                        # 손 끝 (엄지-새끼) 랜드마크만 추출
                        cur_left_hand_end = left_hand_array[[4, 8, 12, 16, 20], :]

            if left_hand:   # 인터페이스에 있는 기능들은 왼손으로 선택 가능

                # 검지 손 끝으로 화면 밑에서 숫자, 문자를 선택하는지 확인
                index_finger = cur_left_hand_end[1]
                if index_finger[1] > (HEIGHT - BOX_HEIGHT)/HEIGHT:
                    if index_finger[0] < 0.5:
                        IS_NUM = True
                    elif index_finger[0] > 0.5:
                        IS_NUM = False
                # 검지 손 끝으로 화면 위에서 del을 선택하는지 확인
                elif index_finger[1] < BOX_HEIGHT / HEIGHT:
                    if index_finger[0] > 0.8:
                        pyautogui.hotkey('backspace')
            
            
            # 각 손가락 끝 속도 계산할 거리 계산
            dist = np.sqrt(np.sum((cur_hand_end - prev_hand_end) ** 2, axis=1))
            cur_time = time.time()

            # TIME_FRAME만큼의 시간이 지날 때마다 손가락 끝 속도 계산하기
            if cur_time - prev_time > TIME_FRAME:
                speed = dist / TIME_FRAME

                # 다섯 손가락중 모든 손가락 속도가 기준치를 넘어서지 않으면(비교적 가만히 있으면) 인식하기
                if any(speed < SPEED_LIMIT):
                    cur_res_final = detect_ML(hand_array)
                    # 쌍자음 인식
                    if cur_res_final in double_const.keys():
                        # TODO: 머신러닝 쌍자음 인식 알고리즘 짜기
                        pass
                # 다음 속도 계산을 위해시간 리셋
                prev_time = time.time()
            # 다음 속도 계산을 위해 손 끝 랜드마크 리셋
            prev_hand_end = cur_hand_end

        else:
            # 손이 인식되지 않으면, 변수들 비우기
            cur_res_final = ""
            landmark_input = deque()

        cur_res_final_list = cur_res_final.split(",")
        if IS_NUM:
            cur_res_final = cur_res_final_list[0]
        else:
            cur_res_final = cur_res_final_list[-1]


        # 이번 프레임에서 인식된 글자가 이전에 인식된 글자와 다르면, 최종 타이핑할 결과에 append하기
        if cur_res_final and cur_res_final != prev_res_final:
            keyboard(cur_res_final)
            final_type_store.append(cur_res_final)
            prev_res_final = cur_res_final

        print(final_type_store, end='\r')

        # 보기 편하게 화면 좌우 반전
        image = cv2.flip(image, 1)

        # 결과 문자 화면에 그리기
        image = myPutText(image, cur_res_final, (500, 50), FONT_SIZE, COLOR)
        # 화면 아래에 한글, 숫자 버튼 표시하는 코드
        cv2.rectangle(
            image,
            (0, HEIGHT - BOX_HEIGHT),
            (int(WIDTH / 2), HEIGHT),
            (0, 255, 0),
            thickness=-1,
        )
        image = myPutText(
            image, "가", (int(WIDTH / 4), HEIGHT - int(BOX_HEIGHT / 2)), 40, (0, 0, 0)
        )
        cv2.rectangle(
            image,
            (int(WIDTH / 2), HEIGHT - BOX_HEIGHT),
            (WIDTH, HEIGHT),
            (0, 0, 255),
            thickness=-1,
        )
        image = myPutText(
            image,
            str(123),
            (int(3 * WIDTH / 4), HEIGHT - int(BOX_HEIGHT / 2)),
            40,
            (0, 0, 0),
        )
        # 삭제 del
        cv2.rectangle(
            image,
            (0, 0),
            (int(WIDTH / 5), BOX_HEIGHT),
            (255,0,0),
            thickness=-1
        )
        image = myPutText(
            image,
            'DEL',
            (int(WIDTH / 20), BOX_HEIGHT / 3),
            40,
            (0, 0, 0)
        )

        # 'q' 누르면 종료
        cv2.imshow("frame", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()