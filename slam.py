import cv2
# import pygame
import sys
from display import Display
from extractor import Extractor

W =1920//2
H=1080//2

# SCREEN USING PYGAME
"""pygame.init()
display = pygame.display.set_mode((W, H)) 
pygame.display.set_caption('SLAM_python') 
surface = pygame.Surface((W,H)).convert()"""


disp=Display(W ,H)
fe = Extractor()

def process_frame(img):
    # sdl based im2vid

    img = cv2.resize(img, (W,H))
    kps, des, matches = fe.extract(img)
    
    for p in kps:
        u,v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img,(u,v), color=(0,255,0), radius = 3)
        # print(f)


 
    disp.paint(img)
 
    """  # surf = pygame.surfarray.make_surface(img.swapaxes(0,1)).convert() # numpy swapaxes
    pygame.pixelcopy.array_to_surface(surface, img.swapaxes(0,1))
    display.blit(surface, (0,0))
    pygame.display.flip()
    # cv2.imshow('image', img)
    # cv2.waitKey(0)"""
    # print(img.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test_vid/test_1.mp4")

    while cap.isOpened():
        ret, frame= cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break