# Import necessary libraries
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

# Load the image
image_path = 'ComfyUI_00075_.png'  # Change this to your image path
orgimg = np.array(Image.open(image_path))

# Perform prediction
results = model.predict(orgimg)

# Convert the original image to PIL format for drawing
pil_image = Image.fromarray(orgimg)

# Create a draw object
draw = ImageDraw.Draw(pil_image)

# Access the results
for result in results:
    bboxes = result.boxes.xyxy.numpy()  # Bounding boxes in (x1, y1, x2, y2) format
    
    # Print the bounding boxes
    print("Bounding Boxes:", bboxes)
    
    # Draw bounding boxes on the image
    for box in bboxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)  # Draw a red rectangle

# Save the modified image with bounding boxes
output_image_path = 'detected_image.jpeg'  # Change this to your desired output path
pil_image.save(output_image_path)

# Display the modified image with bounding boxes using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(pil_image)
plt.axis('off')  # Hide axes
plt.show()

print(f"Detected image saved as: {output_image_path}")