from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import base64
import requests

app = Flask(__name__)

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Caméra
cap = cv2.VideoCapture(0)

# Canvas et dessin
canvas = None
draw_color = (255, 0, 255)
brush_thickness = 10
eraser_thickness = 50
xp, yp = 0, 0

# API Mathpix
MATHPIX_APP_ID = "YOUR_APP_ID"
MATHPIX_APP_KEY = "YOUR_APP_KEY"

def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def gen_frames():
    global canvas, xp, yp, draw_color

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))

                    fingers = fingers_up(hand_landmarks)

                    # Mode efface
                    if fingers[1] and fingers[2]:
                        xp, yp = 0, 0

                    elif fingers[1] and not fingers[2]:
                        cx, cy = lm_list[8]
                        if xp == 0 and yp == 0:
                            xp, yp = cx, cy

                        if draw_color == (0, 0, 0):
                            cv2.line(frame, (xp, yp), (cx, cy), draw_color, eraser_thickness)
                            cv2.line(canvas, (xp, yp), (cx, cy), draw_color, eraser_thickness)
                        else:
                            cv2.line(frame, (xp, yp), (cx, cy), draw_color, brush_thickness)
                            cv2.line(canvas, (xp, yp), (cx, cy), draw_color, brush_thickness)
                        xp, yp = cx, cy

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
            inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inv_canvas)
            frame = cv2.bitwise_or(frame, canvas)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def image_to_latex(image_path):
    with open(image_path, "rb") as image_file:
        img_data = base64.b64encode(image_file.read()).decode()

    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "Content-type": "application/json"
    }
    data = {
        "src": f"data:image/jpeg;base64,{img_data}",
        "formats": ["latex"],
        "ocr": ["math"]
    }

    response = requests.post("https://api.mathpix.com/v3/text", headers=headers, json=data)
    return response.json().get("latex", "Recognition failed.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color/<color>')
def set_color(color):
    global draw_color
    if color == 'pink':
        draw_color = (255, 0, 255)
    elif color == 'blue':
        draw_color = (255, 0, 0)
    elif color == 'green':
        draw_color = (0, 255, 0)
    elif color == 'eraser':
        draw_color = (0, 0, 0)
    return ('', 204)

@app.route('/recognize_math')
def recognize_math():
    global canvas
    cv2.imwrite("static/drawing.png", canvas)
    latex = image_to_latex("static/drawing.png")
    return render_template('template.html', latex_code=latex)

if __name__ == '__main__':
    app.run(debug=True)
