import cv2
import matplotlib.pyplot as plt

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread('ComfyUI_00075_.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

# Save the image with detected faces
output_image_path = 'detected_faces.png'
cv2.imwrite(output_image_path, img)

# Convert the image from BGR to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with detected faces
plt.imshow(img_rgb)
plt.axis('off')  # Hide axis
plt.title('Detected Faces')
plt.show()

print(f"Image saved as: {output_image_path}")
