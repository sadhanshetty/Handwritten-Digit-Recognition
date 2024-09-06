# Importing necessary libraries
import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Set window size and colors
WINDOWSIZEX = 1024
WINDOWSIZEY = 600
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set whether to save images and load the pre-trained model
IMAGESAVE = False
MODEL = load_model("bestmodel.h5")
LABELS = {0: "ZERO", 1: "ONE",
          2: "TWO", 3: "THREE",
          4: "FOUR", 5: "FIVE",
          6: "SIX", 7: "SEVEN",
          8: "EIGHT", 9: "NINE"}

# Initialize Pygame and set up the display surface and font
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
WHILE_INIT = DISPLAYSURF.map_rgb(WHITE)
pygame.display.set_caption("Digit Board")

# Initialize variables for drawing and prediction
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
PREDICT = True

while True:
    for event in pygame.event.get():
        # Handle quitting the program
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        # Handle drawing on the screen with the mouse
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
            pygame.display.update()
        # Handle starting and stopping drawing with the mouse button
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            # Get the bounding box of the drawn digit and extract the image data
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX,number_xcord[-1] + BOUNDRYINC)
            rect_min_Y, rect_max_Y = max(number_ycord[0] - BOUNDRYINC, 0), min(number_ycord[-1] + BOUNDRYINC,WINDOWSIZEX)
            number_xcord = []
            number_ycord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

            # Save the image if desired
            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            # Predict the digit using the pre-trained model and display the result on the screen
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                textsurface = FONT.render(label, True, RED, WHITE)
                textrecobj = textsurface.get_rect()
                textrecobj.left, textrecobj.bottom = rect_min_x, rect_max_Y
                DISPLAYSURF.blit(textsurface, textrecobj)

            # Handle clearing the screen with the "n" key
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

            pygame.display.update()
