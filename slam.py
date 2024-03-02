import cv2
# import pygame
import sys
import sdl2.ext
sdl2.ext.init()
from display import Display

W =1920//2
H=1080//2

# SCREEN USING PYGAME
"""pygame.init()
display = pygame.display.set_mode((W, H)) 
pygame.display.set_caption('SLAM_python') 
surface = pygame.Surface((W,H)).convert()"""

window = sdl2.ext.Window("SLAM_python_pysdl2", size=(W, H), position = (200,0))
window.show()


def process_frame(img):
    # sdl based im2vid
    img = cv2.resize(img, (W,H))
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_QUIT:
            exit(0)
    surf = sdl2.ext.pixels3d(window.get_surface())
    surf[:,:,0:3] = img.swapaxes(0,1)
    window.refresh()

    """
    x=Display(W ,H)
    x.paint(img)
    """

    # cv2.imshow('image', img)

    """  # surf = pygame.surfarray.make_surface(img.swapaxes(0,1)).convert() # numpy swapaxes
    pygame.pixelcopy.array_to_surface(surface, img.swapaxes(0,1))
    display.blit(surface, (0,0))
    pygame.display.flip()
    # cv2.imshow('image', img)
    # cv2.waitKey(0)"""
    # print(img.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test_1.mp4")

    while cap.isOpened():
        ret, frame= cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break