import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
import pygame
import pymunk
# import math
# import re



model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
detection_result = None

HAND_OFFSET_X = -80   # Mueve la pelota a la izquierda de la posición del dedo
HAND_OFFSET_Y = 100   # Mueve la pelota hacia abajo de la posición del dedo

# Coordenadas para la flecha de dirección
arrow_dx = 0.0
arrow_dy = -1.0  # flecha hacia arriba por defecto


# Variable para detectar si la pelota ya fue lanzada
ball_launched = False

tips_id = [4,8,12,16,20]



def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  global detection_result
  detection_result = result


#def draw_landmarks_on_image(rgb_image, detection_result):
#
 # hand_landmarks_list = detection_result.hand_landmarks
  #annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  #for idx in range(len(hand_landmarks_list)):
   # hand_landmarks = hand_landmarks_list[idx]
  

    # Draw the hand landmarks.
    #hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #hand_landmarks_proto.landmark.extend([
     # landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    #])
    # solutions.drawing_utils.draw_landmarks(
    #   annotated_image,
    #   hand_landmarks_proto,
    #   solutions.hands.HAND_CONNECTIONS,
    #   solutions.drawing_styles.get_default_hand_landmarks_style(),
    #   solutions.drawing_styles.get_default_hand_connections_style())

  #return annotated_image

#--------------------------------------------------------------------------------------------------------------------------

# Configuración de Pygame
pygame.init()
WIDTH, HEIGHT = 960, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Configuración de Pymunk
space = pymunk.Space()
space.gravity = (0, 400) # Añadimos gravedad para el lanzamiento

# Crear un círculo en Pymunk que se moverá con la mano
ball_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
ball_body.position = (320, 400)  # zona baja
ball_shape = pymunk.Circle(ball_body, 18)
space.add(ball_body, ball_shape)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
  cap = cv2.VideoCapture(0)
  running = True
  while cap.isOpened() and running:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image,1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, frame_timestamp_ms)
    if detection_result is not None:
      # image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      
      #image = draw_bb_with_letter(image,detection_result,'A')
      if len(detection_result.hand_landmarks) > 0:
        landmarks = detection_result.hand_landmarks[0]
        # Obtener coordenadas del punto 8 (índice)
        index_finger_tip = landmarks[8]
        index_finger_mcp = landmarks[5]  # nudillo del índice

        palm = landmarks[0]

        # Distancia dedo–palma
        dx = index_finger_tip.x - palm.x
        dy = index_finger_tip.y - palm.y
        distance = (dx**2 + dy**2) ** 0.5

        hand_closed = distance < 0.11


        # Vector de dirección
        dir_x = index_finger_tip.x - index_finger_mcp.x
        dir_y = index_finger_tip.y - index_finger_mcp.y

        # Hacer la flecha en sí
        ARROW_LENGTH = 120

        if not ball_launched:
          # Convertir dirección a coordenadas de pantalla
          dir_screen_x = dir_x * WIDTH
          dir_screen_y = dir_y * HEIGHT

          # Normalizar vector
          length = (dir_screen_x**2 + dir_screen_y**2) ** 0.5
          if length > 0:
            dir_screen_x /= length
            dir_screen_y /= length

          ARROW_LENGTH = 120
          arrow_dx = dir_screen_x * ARROW_LENGTH
          arrow_dy = dir_screen_y * ARROW_LENGTH


        
        # Convertir coordenadas normalizadas a la pantalla de pygame
        screen_x = int(index_finger_tip.x * WIDTH) + HAND_OFFSET_X
        screen_y = int(index_finger_tip.y * HEIGHT) + HAND_OFFSET_Y



        print(index_finger_tip.x,index_finger_tip.y)

        # Limitar zona de control
        min_x, max_x = 50, WIDTH * 0.4
        min_y, max_y = HEIGHT * 0.6, HEIGHT - 50


        clamped_x = max(min_x, min(screen_x, max_x))
        clamped_y = max(min_y, min(screen_y, max_y))

        if not ball_launched:
          ball_body.position = clamped_x, clamped_y


        if hand_closed and not ball_launched:
          ball_launched = True

          ball_body.body_type = pymunk.Body.DYNAMIC

          FORCE = 6_000
          impulse_x = arrow_dx * FORCE
          impulse_y = arrow_dy * FORCE

          if not np.isnan(impulse_x) and not np.isnan(impulse_y):
            ball_body.apply_impulse_at_local_point((impulse_x, impulse_y))


        
    # Avanzar la simulación de Pymunk
    space.step(1 / 60.0)
    # Renderizar el objeto en Pygame
    screen.fill((255, 255, 255))

    if not np.isnan(ball_body.position.x) and not np.isnan(ball_body.position.y):

      ball_x = int(ball_body.position.x)
      ball_y = int(ball_body.position.y)

      pygame.draw.circle(screen, (255, 165, 0), (ball_x, ball_y), int(ball_shape.radius))

      arrow_end_x = int(ball_x + arrow_dx)
      arrow_end_y = int(ball_y + arrow_dy)

      pygame.draw.line(screen, (0, 0, 0), (ball_x, ball_y), (arrow_end_x, arrow_end_y), 4)


    pygame.display.flip()
    clock.tick(60)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
pygame.quit()
  