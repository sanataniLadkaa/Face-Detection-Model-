import cv2
from skimage import io
from skimage.io import imread, imshow, imsave
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = imread('ComfyUI_00075_.png')

# Resize the image (optional)
inv_scale = 1  # Change this value to scale the image
img_resized = resize(img, (img.shape[0] // inv_scale, img.shape[1] // inv_scale), preserve_range=True).astype(np.uint8)

# Convert to BGR format for OpenCV
img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

# Display the original image with detected faces
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.title('Detected Faces')
imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))  # Convert back to RGB for displaying

# Set parameters for HOG
pixels_per_cell = (8, 8)

# Extract HOG features
hogfv, hog_image = hog(img_resized, orientations=9, pixels_per_cell=pixels_per_cell,
                       cells_per_block=(2, 2), visualize=True, channel_axis=-1)

# Rescale HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 5))

# Convert to 8-bit unsigned integer format for saving
hog_image_to_save = (hog_image_rescaled * 255).astype(np.uint8)

# Display the HOG image
plt.subplot(1, 2, 2)
plt.title('HOG Image')
imshow(hog_image_to_save)

# Save the image with detected faces
imsave('detected_faces.png', cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Show the plots
plt.tight_layout()
plt.show()