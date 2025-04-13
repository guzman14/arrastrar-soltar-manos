import cv2
import mediapipe as mp
import math

# Inicialización
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Posición inicial del objeto virtual
obj_x, obj_y = 300, 200
obj_w, obj_h = 100, 100
dragging = False

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # espejo para que sea más intuitivo
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    index_tip = (0, 0)
    thumb_tip = (0, 0)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener coordenadas del pulgar (4) y del índice (8)
            lm_list = hand_landmarks.landmark
            index_tip = (int(lm_list[8].x * w), int(lm_list[8].y * h))
            thumb_tip = (int(lm_list[4].x * w), int(lm_list[4].y * h))

            # Dibujar la mano
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar gesto de agarre
            dist = distance(index_tip, thumb_tip)
            if dist < 40:
                # Si el dedo está sobre el objeto, activar "drag"
                if obj_x < index_tip[0] < obj_x + obj_w and obj_y < index_tip[1] < obj_y + obj_h:
                    dragging = True
            else:
                dragging = False

            # Si está arrastrando, mover el objeto
            if dragging:
                obj_x = index_tip[0] - obj_w // 2
                obj_y = index_tip[1] - obj_h // 2

            # Dibujar puntos de agarre
            cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, thumb_tip, 10, (0, 255, 0), cv2.FILLED)

    # Dibujar el objeto virtual
    cv2.rectangle(img, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), (255, 0, 0), -1)

    # Mostrar imagen
    cv2.imshow("Arrastrar y Soltar", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
