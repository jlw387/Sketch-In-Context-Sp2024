import os
from PIL import Image

def create_empty_files():
    current_dir = r"C:\Users\LabUser\Documents\Sketch-In-Context-Sp2024"
    
    # Define the filename for the empty PNG file
    empty_png_file = os.path.join(current_dir, "listen_img.png")

    # Create a new empty PNG image (1x1 pixel white image)
    empty_image = Image.new("RGB", (1, 1), color="white")
    empty_image.save(empty_png_file)

    print(f"Empty PNG file '{empty_png_file}' created.")

    # Define the filename for the empty text file
    empty_txt_file = os.path.join(current_dir, "listen_export_path.txt")

    # Create an empty text file
    with open(empty_txt_file, "w") as file:
        pass

    print(f"Empty text file '{empty_txt_file}' created.")

if __name__ == "__main__":
    create_empty_files()
