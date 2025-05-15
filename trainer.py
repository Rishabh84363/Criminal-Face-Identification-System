import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataSet"

def getImgID(path):
    # Get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # Create empty face list
    faces = []
    # Create empty ID list
    Ids = []
    # Loop through all the image paths and load the IDs and images
    for imagePath in imagePaths:
        try:
            # Loading the image and converting it to grayscale
            faceImage = Image.open(imagePath).convert('L')
            # Convert the PIL image into a numpy array
            faceNp = np.array(faceImage, 'uint8')
            # Get the ID from the filename (assuming format: user.ID.extension)
            filename = os.path.split(imagePath)[-1]
            Id = int(filename.split(".")[1])
            faces.append(faceNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")
    return Ids, faces

try:
    Ids, faces = getImgID(path)
    print(f"Collected {len(faces)} faces and {len(Ids)} IDs.")
    recognizer.train(faces, np.array(Ids))
    recognizer.write('recognizer\\training_data.yml')
    print("Training complete. Data saved to 'recognizer\\training_data.yml'.")
except Exception as e:
    print(f"An error occurred: {e}")

cv2.destroyAllWindows()
