import csv
import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def preprocess_data(data_dir, output_csv_path, grayscale=True):
    """
    Preprocesses RGB and thermal images in a dataset directory and extracts GLCM features.

    Args:
        data_dir: Root directory containing subdirectories for RGB and thermal images.
        output_csv_path: Path to store the extracted GLCM features as a CSV file.
        grayscale: Flag to convert images to grayscale (True) or keep RGB (False).
    """
    # Create a CSV file to store the extracted features
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row with feature names
        writer.writerow(['contrast', 'homogeneity', 'energy', 'correlation', 'asm',
                         'dissimilarity', 'second_angular_moment', 'variance',
                         'sum_average', 'sum_variance', 'class_label'])

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

                # Extract GLCM features
                features = extract_glcm_features(img)

                # Write features and class label to CSV row
                writer.writerow(features + [class_label])


def extract_glcm_features(image):
    """
    Extracts eleven GLCM features from a grayscale image.

    Args:
        image: Grayscale image as a NumPy array.

    Returns:
        A list containing the eleven GLCM features.
    """
    # Define distance and angles for GLCM calculation
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Feature list to store results
    features = []

    # Extract features for each distance and angle combination
    for d in distances:
        for angle in angles:
            glcm = graycomatrix(image, distances=[d], angles=[angle], levels=256, symmetric=True, normed=True)

            # Extract eleven features from the GLCM
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0],
                graycoprops(glcm, 'ASM')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'ASM')[0, 0],
                np.var(glcm),
                np.sum(glcm * np.arange(2, 2 * 5 + 1, 2)),
                np.sum(glcm * (np.arange(2, 2 * 5 + 1, 2) - np.mean(glcm))) ** 2
            ])

    return features

# Example usage:
data_dir = r"C:\Users\Parth\Desktop\splitted gray data"
output_csv_path = r"C:\Users\Parth\Desktop\glcm_features_splitted.csv"
preprocess_data(data_dir, output_csv_path)
print("CSV file created with GLCM features.")
