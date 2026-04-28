import math
import sys

import pygame

WIDTH, HEIGHT = 800, 500
FPS = 60

BLACK = (0, 0, 0)
BLUE = (0, 120, 255)
WHITE = (255, 255, 255)
GRAY = (30, 30, 30)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Piłka 2D - 360° sterowanie")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

x, y = WIDTH // 2, HEIGHT // 2
radius = 15

angle = 45
speed = 5

vx = math.cos(math.radians(angle)) * speed
vy = math.sin(math.radians(angle)) * speed

paused = False


def update_velocity_from_angle():
    global vx, vy
    rad = math.radians(angle)
    vx = math.cos(rad) * speed
    vy = math.sin(rad) * speed


def draw():
    screen.fill(BLACK)

    pygame.draw.circle(screen, BLUE, (int(x), int(y)), radius)

    text = font.render(
        f"Angle: {angle:.0f}° | Speed: {speed:.1f} | Paused: {paused}", True, WHITE
    )
    screen.blit(text, (10, 10))

    if paused:
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(160)
        overlay.fill(GRAY)
        screen.blit(overlay, (0, 0))

        info1 = font.render("PAUZA - sterowanie:", True, WHITE)
        info2 = font.render("A / D -> zmiana kata (0-360°)", True, WHITE)
        info3 = font.render("W / S -> predkosc", True, WHITE)
        info4 = font.render("SPACE -> wznow", True, WHITE)

        screen.blit(info1, (WIDTH // 2 - 140, HEIGHT // 2 - 60))
        screen.blit(info2, (WIDTH // 2 - 140, HEIGHT // 2 - 30))
        screen.blit(info3, (WIDTH // 2 - 140, HEIGHT // 2))
        screen.blit(info4, (WIDTH // 2 - 140, HEIGHT // 2 + 30))

    pygame.display.flip()


while True:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

            if paused:
                if event.key == pygame.K_a:
                    angle -= 5
                if event.key == pygame.K_d:
                    angle += 5

                angle %= 360

                if event.key == pygame.K_w:
                    speed += 0.5
                if event.key == pygame.K_s:
                    speed = max(0.5, speed - 0.5)

                update_velocity_from_angle()

    if not paused:
        x += vx
        y += vy

        if x - radius <= 0 or x + radius >= WIDTH:
            vx *= -1
            angle = math.degrees(math.atan2(vy, vx))

        if y - radius <= 0 or y + radius >= HEIGHT:
            vy *= -1
            angle = math.degrees(math.atan2(vy, vx))

    draw()
