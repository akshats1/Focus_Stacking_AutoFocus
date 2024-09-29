import io
import time
import numpy as np
import concurrent.futures
import subprocess
import logging
import os
from datetime import datetime
from picamera2 import Picamera2
import cv2
import gpiod
import threading
from PIL import Image 


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
        self.is_camera_running = False
        self.scan_count = 0
        self.ControlPinX = [18, 23, 24, 25]
        self.ControlPinY = [5, 6, 13, 19]
        self.ControlPinZ = [21, 20, 2, 16]
        self.STEPS_PER_MM_X = 10
        self.STEPS_PER_MM_Y = 10
        self.STEPS_PER_MM_Z = 10
        self.delay = 0.0004 # Reduced delay for faster motor movement
        self.seg_right = [
            [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
            [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]
        ]
        self.seg_left = [
            [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0],
            [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]
        ]
        self.setup_gpio()
        self.previous_z = None  # Store previous z reference
        self.z_stack_images = []  # To store z-stack images for focus stacking

    # GPIO Setup (same as before)
    
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
        
    def preprocess_image(self,image):
        """
        Preprocess the input image by converting it to grayscale, 
        applying Gaussian blur for noise reduction, and enhancing contrast.
    
        :param image: Input image (in color, BGR format)
        :return: Preprocessed grayscale image
        """
        
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Optional Step 3: Apply contrast enhancement (if needed)
    # Can use histogram equalization or CLAHE for contrast adjustment
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_image = clahe.apply(blurred_image)
    
        return contrast_image
        
    def variance(self, image):
        """
        Calculate the variance of the input image to assess sharpness.
    
         :param image: Input image (grayscale)
        :return: Variance value indicating sharpness
         """
       # Calculate the Laplacian of the image
        
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
       # Calculate and return the variance
        variance_value = laplacian.var()
        return variance_value


    

    def auto(self):
        start_time_auto = time.perf_counter()
        obj_value = 10
        z_positions = []
        variances = []
        max_variance = 0
        max_z = self.z if self.previous_z is None else self.previous_z  # Start with previous z if available

        # Configure camera for autofocus
        self.configure_camera_for_autofocus()

        step_size = 50 if obj_value == 4 else 25
        max_iterations = 7
        initial_steps = 2  # Number of steps to check in each direction
        threshold = 0.1  # Variance threshold to stop autofocus
        direction = None

        start_time_z_axis = time.perf_counter()
        for i in range(max_iterations):
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            preprocessed_image = self.preprocess_image(image)
            current_variance = self.variance(preprocessed_image)
            variances.append(current_variance)
            z_positions.append(self.z)

            print(f"Iteration {i + 1}: x_axis={self.x}, y_axis={self.y}, z_axis={self.z}, variance={current_variance}")

            if current_variance > max_variance:
                max_variance = current_variance
                max_z = self.z

            # If direction has not been determined, try both directions
            if direction is None:
                if i < initial_steps:
                    command = "zcclk"
                    self.motor_control(command, step_size)
                elif i < initial_steps * 2:
                    command = "zclk"
                    self.motor_control(command, step_size)
                else:
                    # Determine the direction based on initial steps
                    if variances[initial_steps - 1] > variances[-1]:  # Upward direction is better
                        direction = 1
                        print("Direction: Upward")
                    else:  # Downward direction is better
                        direction = -1
                        print("Direction: Downward")

            # Move z-axis based on the determined direction
            if direction is not None:
                command = "zcclk" if direction == 1 else "zclk"
                self.motor_control(command, step_size)

            # Check if variance improvement is below the threshold
            if len(variances) >= 4:
                variance_diff_1 = abs(variances[-1] - variances[-2])
                variance_diff_2 = abs(variances[-3] - variances[-4])
                if variance_diff_1 < threshold and variance_diff_2 < threshold:
                    print("Variance Change below threshold. Stopping autofocus.")
                    break

        # Adjust to the position with the maximum variance
        end_time_z_axis = time.perf_counter()
        print("AutoFocus Z_iteration_Time:", end_time_z_axis - start_time_z_axis, "seconds")

        adjust_steps = self.z - max_z
        command = "zclk" if adjust_steps > 0 else "zcclk"
        self.motor_control(command, abs(adjust_steps))

        # Store the best focus position
        self.previous_z = max_z

        # Capture z-stack images
        self.capture_z_stack(max_z)

        end_time_auto = time.perf_counter()
        print("AutoFocus duration:", end_time_auto - start_time_auto, "seconds")

    # Focus stacking function
    def focus_stack(self, images):
        """Combine z-stack images into one focused image."""
        sharpest_images = []
        for i in range(len(images[0])):
            sharpest = max(images, key=lambda img: cv2.Laplacian(img[i], cv2.CV_64F).var())
            sharpest_images.append(sharpest)
        stacked_image = np.median(np.array(sharpest_images), axis=0).astype(np.uint8)
        return stacked_image

    # Capture z-stack images function
    def capture_z_stack(self, max_z):
        step_size = 10  # Adjust step size for z-stack spacing
        self.z_stack_images = []

        for offset in range(-2, 3):  # Capture 5 images for z-stacking
            z_adjust = max_z + (offset * step_size)
            adjust_steps = self.z - z_adjust
            command = "zclk" if adjust_steps > 0 else "zcclk"
            self.motor_control(command, abs(adjust_steps))

        # Capture the image at this z position
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Check if the image is valid
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                print(f"Failed to capture image at z={z_adjust}")
                continue  # Skip this iteration if the image is invalid

        # Debugging prints
            print(f"Captured image at z={z_adjust}: type={type(image)}, shape={image.shape}")

            self.z_stack_images.append(image)

    # Perform focus stacking on the captured z-stack images
        stacked_image = self.focus_stack(self.z_stack_images)
        #stacked_image_path = f"/media/rasp5/New Volume5/Images_Scan/Test_5/Z_Stack_{datetime.now().strftime('%Y%m%d_%H%M')}/stacked_image.tiff"
        #self.save_image(stacked_image, stacked_image_path)


    # Return to the best focus z position
        adjust_steps = self.z - max_z
        command = "zclk" if adjust_steps > 0 else "zcclk"
        self.motor_control(command, abs(adjust_steps))

    # Return the stacked image
        return stacked_image
        
    def save_image(self, image, image_path):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_pil.save(image_path, format='TIFF')

    def scan(self):
        cur_time = datetime.now()
        dir_path = "/media/rasp5/New Volume5/Images_Scan/Test_5/Z_Stack_{}".format(cur_time.strftime("%Y%m%d_%H%M"))
        subprocess.run(["mkdir", dir_path])

        # Thread pool for saving images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            save_futures = []
            start_time_row_1 = time.perf_counter()

            for i in range(10):
                image_row = []
                drift_factor = 2  # Compensate for drift after each row
                for j in range(10):
                    if j % 3== 0:  # Autofocus less frequently (every 5 images)
                        self.camera.stop()
                        stacked_image=self.auto()  # Perform autofocus and capture z-stack

                    stacked_image_path = os.path.join(dir_path, "stacked_image_{}_{}.tiff".format(i, j))
                    if stacked_image is not None:
                        future = executor.submit(self.save_image, stacked_image, stacked_image_path)
                        save_futures.append(future)

                    image_row.append(stacked_image)  # Store the image for this row

                    self.configure_camera_for_full_resolution()  # Switch to full resolution before capture

                    # Capture the image
                    stream = io.BytesIO()
                    self.camera.capture_file(stream, format='jpeg')
                    stream.seek(0)
                    image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                        print(f"Failed to capture image at (row {i}, col {j})")
                        continue  # Skip saving if the image is invalid

                    image_path = os.path.join(dir_path, "img_{}_{}.tiff".format(i, j))
                    future = executor.submit(self.save_image, image, image_path)
                    save_futures.append(future)

                    # Simulate scanning progress in UI
                    progress = (i * 10 + j + 1) / (10 * 10) * 100
                    print(f"Scanning progress: {progress:.2f}%")

                    # Move the x-axis in alternating directions
                    if i % 2 == 0:
                        start_time = time.perf_counter()
                        self.motor_control("xcclk", 6)
                        end_time = time.perf_counter()
                        print(end_time - start_time, "Duration_x_step")
                    else:
                        self.motor_control("xclk", 6)

                # Move the y-axis after completing the row
                end_time_row_1 = time.perf_counter()
                print("Row scanning duration:", end_time_row_1 - start_time_row_1, "seconds")

                # Introduce a delay after every 2 rows to adjust for drift
                if i % drift_factor == 0 and i > 0:
                    self.motor_control("ycclk", 1)
                else:
                    self.motor_control("ycclk", 6)

            # Ensure all images are saved
            for future in concurrent.futures.as_completed(save_futures):
                future.result()

            print("Scanning complete!")


if __name__ == "__main__":
    microscope = Microscope()
    try:
        microscope.scan()
        print("Scanning process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during scanning: {e}")
        print(f"An error occurred: {e}")
    
        
