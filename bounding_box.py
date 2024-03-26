


import cv2
import numpy as np

# Load the image
image_path = r'C:\Users\pooja\Desktop\Captures\test.jpg'
image = cv2.imread(image_path)

# Define the list of bounding box parameters
bounding_boxes = [
    (1265,578,90,78),
    (1044,444,111,90),
    (1299,381,140,101),
]

# Draw bounding boxes on the image and add center coordinates
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle, thickness=2
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)  # Red circle at center
    cv2.putText(image, f'({cx},{cy})', (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Resize the output image to make it smaller
output_width = 1600  # Set the desired width
scale_factor = output_width / image.shape[1]
output_height = int(image.shape[0] * scale_factor)
output_image = cv2.resize(image, (output_width, output_height))

# Display the resized image with bounding boxes and center coordinates
cv2.imshow("Object Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

