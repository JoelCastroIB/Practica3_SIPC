import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
import pygame
import pymunk

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

detection_result = None

HAND_OFFSET_X = -80
HAND_OFFSET_Y = 100

trajectory = []

charging = False
charge_start_time = 0.0

MIN_FORCE = 1500
MAX_FORCE = 3000
MAX_CHARGE_TIME = 1.5

arrow_dx = 0.0
arrow_dy = -1.0

frozen_arrow_dx = 0.0
frozen_arrow_dy = -1.0

prev_pinch = False
ball_launched = False

score = 0
ball_scored = False

GAME_DURATION = 60
game_start_time = None
game_over = False

restart_button_rect = None

tips_id = [4, 8, 12, 16, 20]

last_ball_y = 0
has_crossed_above_hoop = False

def reset_game():
    global score, ball_scored, game_start_time, game_over
    global ball_launched, trajectory, ball_body, ball_shape, space
    global last_ball_y, has_crossed_above_hoop

    score = 0
    ball_scored = False
    game_start_time = None
    game_over = False
    ball_launched = False
    trajectory.clear()
    last_ball_y = 0
    has_crossed_above_hoop = False

    try:
        space.remove(ball_body, ball_shape)
    except:
        pass

    ball_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    ball_body.position = (320, 400)
    ball_shape = pymunk.Circle(ball_body, 18)
    space.add(ball_body, ball_shape)

    return ball_body, ball_shape

def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detection_result
    detection_result = result

def check_hoop_collision(ball_pos, ball_vel_y):
    global score, ball_scored, has_crossed_above_hoop
    
    hoop_center_x, hoop_center_y = WIDTH - 180, 200
    hoop_radius = 35
    
    ball_x, ball_y = ball_pos
    distance_x = abs(ball_x - hoop_center_x)
    
    if (distance_x < hoop_radius + 10 and
        ball_y < hoop_center_y - 5 and
        not has_crossed_above_hoop and 
        ball_launched):
        has_crossed_above_hoop = True
        return False
    
    if (has_crossed_above_hoop and 
        distance_x < hoop_radius + 5 and
        hoop_center_y - 10 < ball_y < hoop_center_y + 30 and
        ball_vel_y > 0 and
        not ball_scored):
        
        distance_to_center = ((ball_x - hoop_center_x)**2 + (ball_y - hoop_center_y)**2)**0.5
        
        if distance_to_center < hoop_radius:
            score += 2
            ball_scored = True
            has_crossed_above_hoop = False
            print(f"¡CANASTA! Puntos: {score}")
            return True
    
    if ball_y > hoop_center_y + 50:
        has_crossed_above_hoop = False
        
    return False

pygame.init()
WIDTH, HEIGHT = 960, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0, 3000)

ceiling = pymunk.Segment(space.static_body, (0, 30), (WIDTH, 30), 5)
ceiling.elasticity = 0.6
space.add(ceiling)

left_wall = pymunk.Segment(space.static_body, (30, 0), (30, HEIGHT), 5)
left_wall.elasticity = 0.6
space.add(left_wall)

right_wall = pymunk.Segment(space.static_body, (WIDTH - 30, 0), (WIDTH - 30, HEIGHT), 5)
right_wall.elasticity = 0.6
space.add(right_wall)

hoop_radius = 35
hoop_center = (WIDTH - 180, 200)

backboard_top_left = (WIDTH - 130, 100)
backboard_top_right = (WIDTH - 110, 100)
backboard_bottom_left = (WIDTH - 130, 300)
backboard_bottom_right = (WIDTH - 110, 300)

backboard_left = pymunk.Segment(
    space.static_body,
    backboard_top_left,
    backboard_bottom_left,
    5
)
backboard_left.elasticity = 0.8
backboard_left.friction = 0.5
space.add(backboard_left)

backboard_right = pymunk.Segment(
    space.static_body,
    backboard_top_right,
    backboard_bottom_right,
    5
)
backboard_right.elasticity = 0.8
backboard_right.friction = 0.5
space.add(backboard_right)

backboard_top = pymunk.Segment(
    space.static_body,
    backboard_top_left,
    backboard_top_right,
    5
)
backboard_top.elasticity = 0.8
backboard_top.friction = 0.5
space.add(backboard_top)

backboard_bottom = pymunk.Segment(
    space.static_body,
    backboard_bottom_left,
    backboard_bottom_right,
    5
)
backboard_bottom.elasticity = 0.8
backboard_bottom.friction = 0.5
space.add(backboard_bottom)

ball_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
ball_body.position = (320, 400)
ball_shape = pymunk.Circle(ball_body, 18)
ball_shape.elasticity = 0.8
ball_shape.friction = 0.7
space.add(ball_body, ball_shape)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    running = True

    while cap.isOpened() and running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and game_over:
                mouse_pos = pygame.mouse.get_pos()
                if restart_button_rect and restart_button_rect.collidepoint(mouse_pos):
                    ball_body, ball_shape = reset_game()

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        frame_timestamp_ms = int(time.time() * 1000)

        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if detection_result is not None and len(detection_result.hand_landmarks) > 0:

            if game_start_time is None:
                game_start_time = time.time()

            elapsed_time = time.time() - game_start_time
            time_remaining = max(0, GAME_DURATION - elapsed_time)

            if time_remaining <= 0 and not game_over:
                game_over = True
                print(f"\n¡JUEGO TERMINADO! Puntuación final: {score}")

            if not game_over:

                landmarks = detection_result.hand_landmarks[0]

                index_finger_tip = landmarks[8]
                index_finger_mcp = landmarks[5]
                thumb_tip = landmarks[4]

                dx = index_finger_tip.x - thumb_tip.x
                dy = index_finger_tip.y - thumb_tip.y
                pinch_distance = (dx**2 + dy**2)**0.5

                pinch = pinch_distance < 0.04
                just_pinch = pinch and not prev_pinch
                released_pinch = not pinch and prev_pinch
                prev_pinch = pinch

                dir_x = index_finger_tip.x - index_finger_mcp.x
                dir_y = index_finger_tip.y - index_finger_mcp.y

                if not ball_launched:
                    dir_screen_x = dir_x * WIDTH
                    dir_screen_y = dir_y * HEIGHT

                    length = (dir_screen_x**2 + dir_screen_y**2)**0.5
                    if length > 0:
                        dir_screen_x /= length
                        dir_screen_y /= length

                    ARROW_LENGTH = 120
                    arrow_dx = dir_screen_x * ARROW_LENGTH
                    arrow_dy = dir_screen_y * ARROW_LENGTH

                screen_x = int(index_finger_tip.x * WIDTH) + HAND_OFFSET_X
                screen_y = int(index_finger_tip.y * HEIGHT) + HAND_OFFSET_Y

                min_x, max_x = 50, WIDTH * 0.4
                min_y, max_y = HEIGHT * 0.6, HEIGHT - 50

                clamped_x = max(min_x, min(screen_x, max_x))
                clamped_y = max(min_y, min(screen_y, max_y))

                if not ball_launched:
                    ball_body.position = clamped_x, clamped_y

                if just_pinch and not ball_launched:
                    charging = True
                    charge_start_time = time.time()
                    frozen_arrow_dx = arrow_dx
                    frozen_arrow_dy = arrow_dy

                if released_pinch and charging and not ball_launched:
                    charging = False
                    ball_launched = True

                    charge_time = time.time() - charge_start_time
                    charge_time = min(charge_time, MAX_CHARGE_TIME)

                    force_ratio = charge_time / MAX_CHARGE_TIME
                    FORCE = MIN_FORCE + force_ratio * (MAX_FORCE - MIN_FORCE)

                    space.remove(ball_body, ball_shape)

                    mass = 1
                    radius = 18
                    moment = pymunk.moment_for_circle(mass, 0, radius)

                    ball_body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
                    ball_body.position = clamped_x, clamped_y

                    ball_shape = pymunk.Circle(ball_body, radius)
                    ball_shape.elasticity = 0.8
                    ball_shape.friction = 0.7
                    space.add(ball_body, ball_shape)

                    length = (frozen_arrow_dx**2 + frozen_arrow_dy**2)**0.5
                    if length > 0:
                        launch_x = frozen_arrow_dx / length
                        launch_y = frozen_arrow_dy / length
                    else:
                        launch_x, launch_y = 0, -1

                    impulse_x = launch_x * FORCE
                    impulse_y = launch_y * FORCE

                    ball_body.apply_impulse_at_local_point((impulse_x, impulse_y))

                    ball_scored = False
                    has_crossed_above_hoop = False

        if ball_launched and not ball_scored:
            current_y = ball_body.position.y
            vel_y = (current_y - last_ball_y) * 60 if last_ball_y != 0 else 0
            last_ball_y = current_y
            
            check_hoop_collision((ball_body.position.x, ball_body.position.y), vel_y)

        if ball_launched:

            out_of_bounds = (
                ball_body.position.y > HEIGHT + 50 or
                ball_body.position.x < -50 or
                ball_body.position.x > WIDTH + 50
            )

            if out_of_bounds:
                space.remove(ball_body, ball_shape)

                ball_launched = False
                trajectory.clear()

                ball_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                ball_body.position = (320, 400)
                ball_shape = pymunk.Circle(ball_body, 18)
                ball_shape.elasticity = 0.8
                ball_shape.friction = 0.7
                space.add(ball_body, ball_shape)
                
                last_ball_y = 0
                has_crossed_above_hoop = False

        if ball_launched:
            if 0 <= ball_body.position.x <= WIDTH and 0 <= ball_body.position.y <= HEIGHT:
                trajectory.append((ball_body.position.x, ball_body.position.y))

        if len(trajectory) > 200:
            trajectory.pop(0)

        space.step(1 / 60.0)

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

            if charging:
                arrow_end_x = int(ball_x + frozen_arrow_dx)
                arrow_end_y = int(ball_y + frozen_arrow_dy)
            else:
                arrow_end_x = int(ball_x + arrow_dx)
                arrow_end_y = int(ball_y + arrow_dy)

            pygame.draw.line(screen, (0, 0, 0), (ball_x, ball_y), (arrow_end_x, arrow_end_y), 4)

        pygame.draw.line(screen, (100, 100, 100), (0, 30), (WIDTH, 30), 5)
        pygame.draw.line(screen, (100, 100, 100), (30, 0), (30, HEIGHT), 5)
        pygame.draw.line(screen, (100, 100, 100), (WIDTH - 30, 0), (WIDTH - 30, HEIGHT), 5)

        tablero_left = WIDTH - 130
        tablero_top = 100
        tablero_width = 20
        tablero_height = 200
        
        shadow_rect = pygame.Rect(tablero_left + 2, tablero_top + 2, tablero_width, tablero_height)
        pygame.draw.rect(screen, (180, 100, 0), shadow_rect)
        
        backboard_rect = pygame.Rect(tablero_left, tablero_top, tablero_width, tablero_height)
        pygame.draw.rect(screen, (255, 140, 0), backboard_rect)
        pygame.draw.rect(screen, (200, 100, 0), backboard_rect, 4)
        
        square_left = tablero_left + 2
        square_top = tablero_top + 65
        square_width = tablero_width - 4
        square_height = 70
        
        small_square = pygame.Rect(square_left, square_top, square_width, square_height)
        pygame.draw.rect(screen, (255, 180, 100), small_square)
        pygame.draw.rect(screen, (200, 100, 0), small_square, 3)

        hoop_x = int(hoop_center[0])
        hoop_y = int(hoop_center[1])
        hoop_r = int(hoop_radius)

        support_start = (WIDTH - 125, hoop_y)
        support_end = (hoop_x + hoop_r - 10, hoop_y)

        pygame.draw.line(screen, (80, 80, 80), support_start, support_end, 6)
        pygame.draw.line(screen, (120, 120, 120), support_start, (support_end[0], support_end[1] - 1), 2)

        pygame.draw.ellipse(
            screen,
            (150, 60, 0),
            (hoop_x - hoop_r, hoop_y - 10, hoop_r * 2, 20),
            5
        )
        
        pygame.draw.ellipse(
            screen,
            (255, 140, 0),
            (hoop_x - hoop_r + 3, hoop_y - 7, hoop_r * 2 - 6, 14),
            3
        )
        
        pygame.draw.ellipse(
            screen,
            (255, 180, 100),
            (hoop_x - hoop_r + 5, hoop_y - 8, hoop_r * 2 - 10, 4),
            1
        )

        net_color = (240, 240, 240)
        net_shadow = (180, 180, 180)

        net_segments = 6
        net_height = 45

        for i in range(net_segments + 1):
            x_offset = (i / net_segments - 0.5) * hoop_r * 1.8
            x_top = hoop_x + int(x_offset)
            y_top = hoop_y

            x_bottom = hoop_x + int(x_offset * 0.3)
            y_bottom = hoop_y + net_height

            color = net_shadow if i % 2 == 0 else net_color
            pygame.draw.line(screen, color, (x_top, y_top), (x_bottom, y_bottom), 2)

        for j in range(4):
            y_pos = hoop_y + (j + 1) * (net_height / 4)
            width_factor = 1 - (j * 0.2)

            x_left = hoop_x - int(hoop_r * 0.9 * width_factor)
            x_right = hoop_x + int(hoop_r * 0.9 * width_factor)

            pygame.draw.line(screen, net_shadow, (x_left, int(y_pos)), (x_right, int(y_pos)), 2)

        net_bottom_width = int(hoop_r * 0.3)
        pygame.draw.line(
            screen,
            net_shadow,
            (hoop_x - net_bottom_width, hoop_y + net_height),
            (hoop_x + net_bottom_width, hoop_y + net_height),
            2
        )

        font = pygame.font.Font(None, 48)
        score_text = font.render(f"Puntos: {score}", True, (0, 0, 0))
        screen.blit(score_text, (40, 60))

        if game_start_time is not None:
            elapsed_time = time.time() - game_start_time
            time_remaining = max(0, GAME_DURATION - elapsed_time)

            minutes = int(time_remaining // 60)
            seconds = int(time_remaining % 60)

            timer_color = (255, 0, 0) if time_remaining < 10 else (0, 0, 0)
            timer_text = font.render(f"Tiempo: {minutes}:{seconds:02d}", True, timer_color)
            screen.blit(timer_text, (WIDTH - 250, 60))

        if game_over:

            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((20, 20, 40))
            screen.blit(overlay, (0, 0))

            panel_width = 500
            panel_height = 350
            panel_x = (WIDTH - panel_width) // 2
            panel_y = (HEIGHT - panel_height) // 2

            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(screen, (40, 40, 60), panel_rect, border_radius=20)
            pygame.draw.rect(screen, (100, 100, 150), panel_rect, 4, border_radius=20)

            big_font = pygame.font.Font(None, 60)
            game_over_text = big_font.render("¡TIEMPO AGOTADO!", True, (255, 100, 100))
            text_rect = game_over_text.get_rect(center=(WIDTH // 2, panel_y + 70))
            screen.blit(game_over_text, text_rect)

            medium_font = pygame.font.Font(None, 56)
            final_score_text = medium_font.render(f"Puntuación: {score}", True, (255, 215, 0))
            score_rect = final_score_text.get_rect(center=(WIDTH // 2, panel_y + 150))
            screen.blit(final_score_text, score_rect)

            button_width = 280
            button_height = 60
            button_x = (WIDTH - button_width) // 2
            button_y = panel_y + 220

            restart_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            mouse_pos = pygame.mouse.get_pos()
            is_hovering = restart_button_rect.collidepoint(mouse_pos)

            button_color = (80, 200, 120) if is_hovering else (60, 180, 100)
            border_color = (100, 255, 150) if is_hovering else (80, 220, 120)

            pygame.draw.rect(screen, button_color, restart_button_rect, border_radius=15)
            pygame.draw.rect(screen, border_color, restart_button_rect, 4, border_radius=15)

            button_font = pygame.font.Font(None, 40)
            button_text = button_font.render(" VOLVER A JUGAR", True, (255, 255, 255))
            button_text_rect = button_text.get_rect(center=restart_button_rect.center)
            screen.blit(button_text, button_text_rect)

            small_font = pygame.font.Font(None, 28)
            esc_text = small_font.render("Presiona ESC para salir", True, (150, 150, 170))
            esc_rect = esc_text.get_rect(center=(WIDTH // 2, panel_y + panel_height - 30))
            screen.blit(esc_text, esc_rect)

        pygame.display.flip()
        clock.tick(60)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            running = False
            break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
