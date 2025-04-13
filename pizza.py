import cv2
import mediapipe as mp
import math

# Cargar imágenes
pizza_img = cv2.imread("images/pizza.png", cv2.IMREAD_UNCHANGED)
personaje_img = cv2.imread("images/imagen.png", cv2.IMREAD_UNCHANGED)

# Redimensionar imágenes si son muy grandes
pizza_img = cv2.resize(pizza_img, (100, 100))
personaje_img = cv2.resize(personaje_img, (120, 150))

# Posición fija del personaje
personaje_x, personaje_y = 450, 250

# Inicializar detección de manos
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Posición de la pizza
pizza_x, pizza_y = 200, 150
dragging = False

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]

    # Si la imagen tiene canal alfa (transparencia)
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a)) / 255.0
        roi = bg[y:y+h, x:x+w]
        blended = roi * (1 - mask) + overlay_rgb * mask
        bg[y:y+h, x:x+w] = blended.astype("uint8")
    else:
        # Si NO tiene canal alfa, simplemente la sobrepone
        bg[y:y+h, x:x+w] = overlay

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    index_tip = (0, 0)
    thumb_tip = (0, 0)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dist = distance(index_tip, thumb_tip)

            if dist < 40 and index_tip[0] in range(pizza_x, pizza_x + 100) and index_tip[1] in range(pizza_y, pizza_y + 100):
                dragging = True
            elif dist >= 40:
                dragging = False

            if dragging:
                pizza_x = index_tip[0] - 50
                pizza_y = index_tip[1] - 50

    # Mostrar personaje
    overlay_image(frame, personaje_img, personaje_x, personaje_y)

    # Mostrar pizza
    overlay_image(frame, pizza_img, pizza_x, pizza_y)

    # Verificar si la pizza está cerca del personaje
    center_pizza = (pizza_x + 50, pizza_y + 50)
    center_personaje = (personaje_x + 60, personaje_y + 75)

    if distance(center_pizza, center_personaje) < 80:
        cv2.putText(frame, "¡Pizza entregada!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Arrastrar Pizza al Personaje", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
