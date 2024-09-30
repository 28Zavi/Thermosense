import os
import cv2

def convert_rgb_to_grayscale(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            # Read the image in RGB format
            image = cv2.imread(os.path.join(input_folder, file))

            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image to the output folder
            cv2.imwrite(os.path.join(output_folder, file), grayscale_image)

    print("Conversion completed.")

# Example usage:
input_folder = r"C:\Users\Parth\Desktop\data\og data\thermal\overripe"
output_folder = r"C:\Users\Parth\Desktop\gray data\thermal\overripe"
convert_rgb_to_grayscale(input_folder, output_folder)
