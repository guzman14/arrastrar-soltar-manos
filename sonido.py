import cv2
import mediapipe as mp
import math
import pygame
import random

# Inicializar Pygame y los sonidos
pygame.mixer.init()

# Cargar sonidos
grab_sound = pygame.mixer.Sound("sounds/grab_sound.mp3")  # Sonido cuando se agarra un objeto
release_sound = pygame.mixer.Sound("sounds/release_sound.mp3")  # Sonido cuando se suelta un objeto

# Inicialización de cámara y detección de manos
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Clase para los objetos virtuales
class VirtualObject:
    def __init__(self, x, y, w, h, color):
        self.x, self.y = x, y
        self.w, self.h = w, h  # Corregir esta línea
        self.color = color
        self.dragging = False

    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)

    def is_inside(self, px, py):
        return self.x < px < self.x + self.w and self.y < py < self.y + self.h

    def move_to(self, px, py):
        self.x = px - self.w // 2
        self.y = py - self.h // 2

# Crear una lista de objetos (con colores aleatorios)
objects = [VirtualObject(random.randint(100, 400), random.randint(100, 400), 100, 100, 
                        (random.randint(0,255), random.randint(0,255), random.randint(0,255))) for _ in range(3)]

# Función para medir la distancia entre dos puntos
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Bucle principal
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    index_tip = (0, 0)
    thumb_tip = (0, 0)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dist = distance(index_tip, thumb_tip)

            for obj in objects:
                # Si los dedos están juntos y tocando el objeto
                if dist < 40 and obj.is_inside(index_tip[0], index_tip[1]):
                    if not obj.dragging:  # Si no se está arrastrando, se comienza
                        grab_sound.play()  # Reproducir sonido de agarre
                    obj.dragging = True
                elif dist >= 40:
                    if obj.dragging:  # Si se suelta el objeto
                        release_sound.play()  # Reproducir sonido de soltado
                    obj.dragging = False

                if obj.dragging:
                    obj.move_to(index_tip[0], index_tip[1])

    # Dibujar objetos
    for obj in objects:
        obj.draw(img)

    # Dibujar puntos de dedos
    cv2.circle(img, index_tip, 10, (0, 255, 255), -1)
    cv2.circle(img, thumb_tip, 10, (0, 255, 255), -1)

    cv2.imshow("Arrastrar Múltiples Objetos con Sonidos", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
