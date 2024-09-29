import serial
import cv2
from time import sleep
from datetime import datetime
import subprocess
import logging
from picamera2 import Picamera2, Preview
from PIL import Image
import io
import numpy as np
import os
import time
import threading
import concurrent.futures  # For parallel image saving
import gpiod

# Initialize global variables
x_axis = 0
y_axis = 0
z_axis = 0
camera_lock = threading.Lock()
start_time_auto = 0

class Microscope:
    def __init__(self):
        # Initialize camera and pins
        self.camera = Picamera2()
        self.x = 0
        self.y = 0
        self.z = 4000
        self.start_time_auto = 0
        self.end_time_auto = 0
        self.is_camera_running = False  # Flag to track camera state
        self.scan_count = 0
        self.ControlPinX = [18, 23, 24, 25]
        self.ControlPinY = [5, 6, 13, 19]
        self.ControlPinZ = [21, 20, 2, 16]
        self.STEPS_PER_MM_X = 10
        self.STEPS_PER_MM_Y = 10
        self.STEPS_PER_MM_Z = 10
        self.delay = 0.0004  # Reduced delay for faster motor movement
        self.seg_right = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1]
        ]
        self.seg_left = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1]
        ]
        self.setup_gpio()

    def setup_gpio(self):
        # Initialize GPIO
        self.chip = gpiod.Chip('gpiochip0')
        self.lines_x = [self.chip.get_line(pin) for pin in self.ControlPinX]
        self.lines_y = [self.chip.get_line(pin) for pin in self.ControlPinY]
        self.lines_z = [self.chip.get_line(pin) for pin in self.ControlPinZ]
        self.ms_line = self.chip.get_line(17)
        self.ms_line.request(consumer='stepper', type=gpiod.LINE_REQ_DIR_OUT)
        self.ms_line.set_value(0)
        for line in self.lines_x + self.lines_y + self.lines_z:
            try:
                line.request('stepper_motor', gpiod.LINE_REQ_DIR_OUT, 0)
            except OSError as e:
                print(f"Error requesting line: {e}")

    def move_x(self, forward, steps):
        direction = self.seg_right if forward else self.seg_left
        self.run_motor(self.lines_x, direction, steps)

    def move_y(self, forward, steps):
        direction = self.seg_right if forward else self.seg_left
        self.run_motor(self.lines_y, direction, steps)

    def move_z(self, upward, steps):
        direction = self.seg_left if upward else self.seg_right
       
        self.run_motor(self.lines_z, direction, steps)

    def run_motor(self, lines, direction, steps):
        steps = int(steps)
        for _ in range(steps):
            for halfstep in range(8):
                for pin in range(4):
                    lines[pin].set_value(direction[halfstep][pin])
                    time.sleep(self.delay)

    def motor_control(self, command, steps):
        try:
            if command.startswith("xclk"):
                self.move_x(forward=True, steps=steps)
                self.x -= steps
            elif command.startswith("xcclk"):
                self.move_x(forward=False, steps=steps)
            elif command.startswith("yclk"):
                self.move_y(forward=True, steps=steps)
            elif command.startswith("ycclk"):
                self.move_y(forward=False, steps=steps)
            elif command.startswith("zclk"):
                self.z -= steps
                self.move_z(upward=False, steps=steps)  # Downward movement
            elif command.startswith("zcclk"):
                self.z += steps
                self.move_z(upward=True, steps=steps)  # Upward movement
            elif command == "init":
                self.home_all_axes()
            elif command == "status":
                print(self.check_endstops())
            else:
                print("Unknown command")
        except (IndexError, ValueError) as e:
            print(f"Error processing command: {e}")

    def configure_camera_for_autofocus(self):
        if self.is_camera_running:
            self.camera.stop()
            self.is_camera_running = False
        preview_config = self.camera.create_preview_configuration(
            main={"size": (320, 240)}, raw={"format": "SBGGR10_CSI2P"}
        )
        self.camera.configure(preview_config)
        self.camera.start(show_preview=False)
        self.is_camera_running = True

    def configure_camera_for_full_resolution(self):
        if self.is_camera_running:
            self.camera.stop()
            self.is_camera_running = False
        full_res_config = self.camera.create_still_configuration(
            main={"size": (4056, 3040)}, raw={"format": "SBGGR10_CSI2P"}
        )
        self.camera.configure(full_res_config)
        self.camera.start(show_preview=False)
        self.is_camera_running = True
    
    
    def focus_stacking(self, images):
        # Convert images to grayscale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        # Calculate the sharpness (Laplacian variance) of each pixel
        sharpness = np.stack([cv2.Laplacian(img, cv2.CV_64F) for img in gray_images], axis=-1)
        max_sharpness_index = np.argmax(sharpness, axis=-1)

        # Create an empty image for the focus-stacked result
        stacked_image = np.zeros_like(images[0])

        # For each pixel, choose the value from the image with the highest sharpness
        for i in range(stacked_image.shape[0]):
            for j in range(stacked_image.shape[1]):
                stacked_image[i, j] = images[max_sharpness_index[i, j]][i, j]

        return stacked_image
    def save_image(self, image, image_path):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_pil.save(image_path, format='TIFF')
    
    
    def auto(self, obj_value):
        z_positions = []
        variances = []
        max_variance = 0
        max_z = self.z
        previous_z = self.z  # Initialize previous_z

        # Configure camera for autofocus (low-resolution mode)
        self.configure_camera_for_autofocus()
        
        # Define the step size based on the objective value
        if obj_value == 4:
            step_size = 50
        elif obj_value == 10:
            step_size = 20
        elif obj_value == 40:
            step_size = 5
        
        # Autofocus loop to find the maximum variance
        for i in range(7):
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Calculate variance
            current_variance = self.variance(image)
            variances.append(current_variance)
            z_positions.append(self.z)

        # Update the max variance and corresponding z position
            if current_variance > max_variance:
                max_variance = current_variance
                max_z = self.z

        # Move the Z-axis upward if the variance improves
            if i < 6:  # Allow for upward movement for the first 6 iterations
                self.move_z(upward=True, steps=step_size)
                self.z += step_size
                sleep(1)  # Pause to allow motor to stabilize

            # Capture again to check variance after moving up
                stream = io.BytesIO()
                self.camera.capture_file(stream, format='jpeg')
                stream.seek(0)
                image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                current_variance_up = self.variance(image)

                if current_variance_up > max_variance:
                    max_variance = current_variance_up
                    max_z = self.z

            # Move downward to check for variance improvement
                self.move_z(upward=False, steps=step_size)
                self.z -= step_size
                sleep(1)  # Pause to allow motor to stabilize

            # Capture again to check variance after moving down
                stream = io.BytesIO()
                self.camera.capture_file(stream, format='jpeg')
                stream.seek(0)
                image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                current_variance_down = self.variance(image)

                if current_variance_down > max_variance:
                    max_variance = current_variance_down
                    max_z = self.z

        # Stop if variance does not improve for both upward and downward movements
            if current_variance <= max_variance and previous_z == self.z:
                break  # Stop if no change in z position and variance doesn't improve

            previous_z = self.z  # Update previous z position

    # Capture 5 images symmetrically around max_z for focus stacking
        z_steps = [-2, -1, 0, 1, 2]  # 5 positions around max_z
        images = []

        for step in z_steps:
            adjust_steps = (self.z - max_z) + (step * step_size)

            if adjust_steps > 0:
                self.move_z(upward=False, steps=adjust_steps)  # Move downward
                self.z -= adjust_steps 
            else:
                self.move_z(upward=True, steps=abs(adjust_steps))  # Move upward
                self.z += abs(adjust_steps) 
            sleep(1)  # Pause for motor to stabilize

        # Capture image at the new z position
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            images.append(image)


       
       
       
        sleep(1)
        print(variances)
        print(z_positions)
        print(f"Max Z: {max_z}, Max Variance: {max_variance}")
        return max_z

    def variance(self, image):
        """Calculate the variance of the Laplacian of an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def z_stack(self, z_ref):
        step_size = 15
        if self.z != z_ref:
            adjust_steps = abs(self.z - z_ref)
            if self.z > z_ref:
                self.move_z(upward=False, steps=adjust_steps)  # Move downward
            else:
                self.move_z(upward=True, steps=adjust_steps)  # Move upward

        z_steps = [-2, -1, 0, 1, 2]  # 5 positions around z_ref
        images = []
        z_positions = []

        for step in z_steps:
            # Adjust the z position
            adjust_steps = step * step_size
            if adjust_steps > 0:
                self.move_z(upward=True, steps=adjust_steps)
                
            else:
                self.move_z(upward=False, steps=abs(adjust_steps))

            sleep(1)  # Pause for motor stabilization

            # Capture image at the current z position
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            images.append(image)
            z_positions.append(self.z)

        return images, z_positions
        
        
    def scan(self):
        cur_time = datetime.now()
        dir_path = "/media/rasp5/New Volume5/Images_Scan/Test_5/Z_Stack_{}".format(cur_time.strftime("%Y%m%d_%H%M"))
        subprocess.run(["mkdir", dir_path])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            save_futures = []
            start_time_row_1 = time.perf_counter()
            for i in range(10):
                image_row = []
                for j in range(8):
                    if j % 2 == 0:
                        self.camera.stop()
                        z_ref = self.auto(10)
                        # Save the final focus-stacked image
                        self.configure_camera_for_full_resolution()

                # Capture z-stack images at z_ref
                    images, z_positions = self.z_stack(z_ref=z_ref)
                    image_row.append((images, z_positions))

               
                 
                # Perform focus stacking on the captured z-stack images
                    stacked_image = self.focus_stacking(images)

               
                    stacked_file_name = f"stacked_image_{i}_{j}.tiff"
                    stacked_file_path = os.path.join(dir_path, stacked_file_name)
                    save_futures.append(executor.submit(self.save_image, stacked_image, stacked_file_path))

                # Move x forward to the next position
                    if i % 2 == 0:
                        self.move_x(forward=False, steps=10)
                    # self.scan_count += 1
                    else:
                        self.move_x(forward=True, steps=10)

            
                self.move_y(forward=True, steps=12)

        # Wait for all save operations to complete
        concurrent.futures.wait(save_futures)
        end_time_row_1 = time.perf_counter()
        print(f"scan completed in {end_time_row_1 - start_time_row_1:.2f} seconds")


if __name__ == "__main__":
    microscope = Microscope()
    try:
        microscope.scan()
        print("Scanning process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during scanning: {e}")
        print(f"An error occurred: {e}")
    
        
