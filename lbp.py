import csv
import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern

def preprocess_data_lbp(data_dir, output_csv_path, grayscale=True):
    """
    Preprocesses RGB and thermal images in a dataset directory and extracts LBP features.

    Args:
        data_dir: Root directory containing subdirectories for RGB and thermal images.
        output_csv_path: Path to store the extracted LBP features as a CSV file.
        grayscale: Flag to convert images to grayscale (True) or keep RGB (False).
    """
    # Create a CSV file to store the extracted features
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row with feature names
        writer.writerow(['lbp_{}'.format(i) for i in range(256)] + ['class_label'])

        # List of directories to process
        directories = [
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\rgb\\train\\ripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\rgb\\train\\unripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\rgb\\train\\overripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\rgb\\test\\ripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\rgb\\test\\unripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\rgb\\test\\overripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\thermal\\train\\ripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\thermal\\train\\unripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\thermal\\train\\overripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\thermal\\test\\ripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\thermal\\test\\unripe',
            'C:\\Users\\Parth\\Desktop\\splitted gray data\\thermal\\test\\overripe',
        ]

        # Loop through the directories
        for directory in directories:
            class_label = os.path.basename(os.path.dirname(directory))
            for filename in os.listdir(directory):
                image_path = os.path.join(directory, filename)
                img = cv2.imread(image_path)

                # Grayscale conversion if specified
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Extract LBP features
                features = extract_lbp_features(img)

                # Write features and class label to CSV row
                writer.writerow(features + [class_label])


def extract_lbp_features(image):
    """
    Extracts Local Binary Pattern (LBP) features from a grayscale image.

    Args:
        image: Grayscale image as a NumPy array.

    Returns:
        A list containing the LBP histogram as features.
    """
    # Compute LBP
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

    # Calculate histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 257), range=(0, 256))

    return hist.tolist()

# Example usage:
data_dir_lbp = r"C:\Users\Parth\Desktop\splitted gray data"
output_csv_path_lbp = r"C:\Users\Parth\Desktop\lbp_features_splitted.csv"
preprocess_data_lbp(data_dir_lbp, output_csv_path_lbp)
print("CSV file created with LBP features.")
