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

trajectory = []

# Coordenadas para la flecha de dirección
arrow_dx = 0.0
arrow_dy = -1.0  # flecha hacia arriba por defecto

# Variables globales para congelar posición ante lanzamiento
frozen_arrow_dx = 0.0
frozen_arrow_dy = -1.0

# Detectar el momento antes de la pinza
prev_pinch = False

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
space.gravity = (0, 10000) # Añadimos gravedad para el lanzamiento

# Suelo
floor = pymunk.Segment(space.static_body, (0, HEIGHT - 30), (WIDTH, HEIGHT - 30), 5)
floor.elasticity = 0.6
floor.friction = 0.8
space.add(floor)

# Pared izquierda
left_wall = pymunk.Segment(space.static_body, (30, 0), (30, HEIGHT), 5)
left_wall.elasticity = 0.6
space.add(left_wall)

# Pared derecha
right_wall = pymunk.Segment(space.static_body, (WIDTH - 30, 0), (WIDTH - 30, HEIGHT), 5)
right_wall.elasticity = 0.6
space.add(right_wall)


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

        thumb_tip = landmarks[4]

        dx = index_finger_tip.x - thumb_tip.x
        dy = index_finger_tip.y - thumb_tip.y
        pinch_distance = (dx**2 + dy**2) ** 0.5

        pinch = pinch_distance < 0.04

        just_pinch = pinch and not prev_pinch
        prev_pinch = pinch

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


        if just_pinch and not ball_launched:
          ball_launched = True

          # Congelar dirección
          frozen_arrow_dx = arrow_dx
          frozen_arrow_dy = arrow_dy

          # Eliminar cuerpo cinemático
          space.remove(ball_body, ball_shape)

          # Crear cuerpo dinámico CON masa
          mass = 1
          radius = 18
          moment = pymunk.moment_for_circle(mass, 0, radius)

          ball_body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
          ball_body.position = clamped_x, clamped_y

          ball_shape = pymunk.Circle(ball_body, radius)
          space.add(ball_body, ball_shape)

          # Normalizar vector de lanzamiento
          length = (frozen_arrow_dx**2 + frozen_arrow_dy**2) ** 0.5
          if length > 0:
            launch_x = frozen_arrow_dx / length
            launch_y = frozen_arrow_dy / length
          else:
            launch_x, launch_y = 0, -1


          FORCE = 20
          impulse_x = frozen_arrow_dx * FORCE
          impulse_y = frozen_arrow_dy * FORCE

          ball_body.apply_impulse_at_local_point((impulse_x, impulse_y))

    # Avanzar la simulación de Pymunk
    space.step(1 / 60.0)

    if ball_launched:
      if (
        0 <= ball_body.position.x <= WIDTH and
        0 <= ball_body.position.y <= HEIGHT
      ):
        trajectory.append((ball_body.position.x, ball_body.position.y))

    # Limitar tamaño del rastro
    if len(trajectory) > 200:
      trajectory.pop(0)

    # Renderizar el objeto en Pygame
    screen.fill((255, 255, 255))

    for p in trajectory:
      if not np.isnan(p[0]) and not np.isnan(p[1]):
        pygame.draw.circle(
            screen,
            (200, 200, 200),
            (int(p[0]), int(p[1])),
            2
        )


    if not np.isnan(ball_body.position.x) and not np.isnan(ball_body.position.y):

      ball_x = int(ball_body.position.x)
      ball_y = int(ball_body.position.y)

      pygame.draw.circle(screen, (255, 165, 0), (ball_x, ball_y), int(ball_shape.radius))

      arrow_end_x = int(ball_x + arrow_dx)
      arrow_end_y = int(ball_y + arrow_dy)

      pygame.draw.line(screen, (0, 0, 0), (ball_x, ball_y), (arrow_end_x, arrow_end_y), 4)

    # Dibujar suelo
    pygame.draw.line(screen, (100, 100, 100), (0, HEIGHT - 30), (WIDTH, HEIGHT - 30), 5)

    # Dibujar paredes
    pygame.draw.line(screen, (100, 100, 100), (30, 0), (30, HEIGHT), 5)
    pygame.draw.line(screen, (100, 100, 100), (WIDTH - 30, 0), (WIDTH - 30, HEIGHT), 5)

    pygame.display.flip()
    clock.tick(60)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
pygame.quit()
  