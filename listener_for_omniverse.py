# import os
# import time
# import subprocess
# from PIL import Image
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# import threading

# current_dir = r"C:\Users\LabUser\Documents\Sketch-In-Context-Sp2024"
# # # Function to create the empty PNG and text files
# # def create_files():
# #     # Define the filename for the empty PNG file
# #     empty_png_file = os.path.join(current_dir, "listen_img.png")

# #     # Create a new empty PNG image (1x1 pixel white image)
# #     empty_image = Image.new("RGB", (1, 1), color="white")
# #     empty_image.save(empty_png_file)

# #     print(f"Empty PNG file '{empty_png_file}' created.")

# #     # Define the filename for the empty text file
# #     empty_txt_file = os.path.join(current_dir, "listen_export_path.txt")

# #     # Create an empty text file
# #     with open(empty_txt_file, "w") as file:
# #         pass

# #     print(f"Empty text file '{empty_txt_file}' created.")

# # Custom event handler for file changes
# class FileChangeHandler(FileSystemEventHandler):
#     def on_modified(self, event):
#         print(event)
#         if event.src_path == empty_png_file or event.src_path == empty_txt_file:
#             print(f"{event.src_path} has been modified")
#             with open(empty_txt_file, "r") as file:
#                 # Read content from the file
#                 export_path = file.read()
#             subprocess.run(["python", "sdf_to_mesh.py", empty_png_file, export_path])

# # Start file monitoring
# def start_file_monitoring():
#     event_handler = FileChangeHandler()
#     observer = Observer()
#     observer.schedule(event_handler, path=".", recursive=False)
#     observer.start()
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# # Create threads for file creation and file monitoring
# # file_creation_thread = threading.Thread(target=create_files)
# file_monitoring_thread = threading.Thread(target=start_file_monitoring)

# # # Start the file creation thread
# # file_creation_thread.start()

# # # Wait for the file creation thread to finish
# # file_creation_thread.join()

# # Start file monitoring in a separate process
# subprocess.Popen(["python", os.path.join(current_dir, "listener_for_omniverse.py")])

import os
import time
from watchdog.observers import Observer
import subprocess
from watchdog.events import FileSystemEventHandler

current_dir = r"C:\Users\LabUser\Documents\Sketch-In-Context-Sp2024"
empty_png_file = os.path.join(current_dir, "listen_img.png")
empty_txt_file = os.path.join(current_dir, "listen_export_path.txt")

# Custom event handler for file changes
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, directory_path):
        super().__init__()
        self.directory_path = directory_path

    def on_modified(self, event):
        print(f'event type: {event.event_type}  path : {event.src_path}')
        if event.src_path[2:] in empty_png_file or event.src_path[2:] in empty_txt_file:
            print(f"{event.src_path} has been modified")
            with open(empty_txt_file, "r") as file:
                # Read content from the file
                export_path = file.read()
            sdf_to_mesh_path = os.path.join(current_dir, "sdf_to_mesh.py")
            subprocess.run(["python", sdf_to_mesh_path, empty_png_file, export_path])

# Start file monitoring
def start_file_monitoring():
    event_handler = FileChangeHandler(current_dir)
    observer = Observer()
    observer.schedule(event_handler, path=current_dir, recursive=True)
    observer.start()
    try:
        while True:
            # Read File

            # If File ! Empty:
                # Empty File

                # Do Network Thing

                # Save OBJ File
            # Else
            time.sleep(0.25)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_file_monitoring()

