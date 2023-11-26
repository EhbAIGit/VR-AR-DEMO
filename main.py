import os
import pygame
from canny_edge_detection import detecting_canny_edge
from classify_images import classify_images
from openpose import openpose
from pose_module import make_pose

pygame.init()
info = pygame.display.Info()
width = info.current_w
height = info.current_h
SCREEN_WIDTH, SCREEN_HEIGHT, MARGIN = width, (height - 50), 100
FPS = 30

class GUIApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Lab EHB")

        self.clock = pygame.time.Clock()

        self.bg_color = (255, 255, 255)  # Initial background color

        file_name = "AR_background.png"
        # Load the splash image
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the file in the same directory as the script
        file_path = os.path.join(script_dir, file_name)
        self.splash_image = pygame.image.load(file_path)
        self.splash_image = pygame.transform.scale(self.splash_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Create a font object for displaying text
        self.font = pygame.font.Font(None, 36)

        self.show_splash_screen()
        self.running = True
        self.create_buttons()

    def create_buttons(self):
        self.buttons = []

        button_width = 350
        button_height = 60
        # Define button properties (position, size, text)
        image_classification_button = {'position': (MARGIN, MARGIN),
                   'size': (button_width, button_height),
                   'text': 'Image Classification',
                   'function': self.image_class_pressed}

        canny_edges_detection_button = {'position': (MARGIN, MARGIN + 70),
                   'size': (button_width, button_height),
                   'text': 'Canny Edges Detection',
                   'function': self.canny_edges_pressed}

        openpose_button = {'position': (MARGIN, MARGIN + 140),
                   'size': (button_width, button_height),
                   'text': 'Pose Estimation (OpenPose)',
                   'function': self.openpose_pressed}

        mediapipe_button = {'position': (MARGIN, MARGIN + 210),
                   'size': (button_width, button_height),
                   'text': 'Pose Estimation (MediaPipe)',
                   'function': self.openpose_pressed}

        self.buttons.extend([image_classification_button, canny_edges_detection_button, openpose_button, mediapipe_button])


        num_buttons = len(self.buttons)

        # Calculate the total height occupied by buttons
        total_buttons_height = num_buttons * button_height

        # Calculate the starting position for the first button to center them vertically
        start_y = (SCREEN_HEIGHT - total_buttons_height) // 2

        for i, button_data in enumerate(self.buttons):
            position = ((SCREEN_WIDTH - button_width) // 2, start_y + i * (button_height + 10))
            button_data['position'] = position
    def image_class_pressed(self):
        classify_images()

    def canny_edges_pressed(self):
        detecting_canny_edge()

    def openpose_pressed(self):
        openpose()
    def mediapipe_pressed(self):
        make_pose()
    def show_splash_screen(self):
        self.screen.blit(self.splash_image, (0, 0))
        pygame.display.flip()

        pygame.time.delay(3000)  # Display splash screen for 3 seconds

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    for button in self.buttons:
                        rect = pygame.Rect(button['position'], button['size'])
                        if rect.collidepoint(mouse_pos):
                            button['function']()

            for button in self.buttons:
                pygame.draw.rect(self.screen, (0, 0, 0), (button['position'], button['size']))
                text = self.font.render(button['text'], True, (255, 255, 255))
                self.screen.blit(text, (button['position'][0] + 10, button['position'][1] + 10))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()




if __name__ == "__main__":
    app = GUIApp()
    app.run()