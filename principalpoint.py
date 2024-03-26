import cv2

# Load your image (replace 'your_image.jpg' with your image file path)
image = cv2.imread(r"C:\Users\pooja\Desktop\Captures\circles.jpg")

# Resize the image to a width of 1600 pixels (maintaining aspect ratio)
new_width = 1600
aspect_ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * aspect_ratio)
resized_image = cv2.resize(image, (new_width, new_height))

# Get the new image dimensions (height and width)
height, width, _ = resized_image.shape

# Calculate the center pixel coordinates (u, v)
u = width // 2
v = height // 2

# Set the color for marking the center (in BGR format, here using red)
mark_color = (0, 0, 255)  # Red in BGR

# Draw a circle or point to mark the center
radius = 5  # You can adjust the size of the marker
thickness = -1  # Use -1 to fill the circle

# Draw a circle at the center of the image
cv2.circle(resized_image, (u, v), radius, mark_color, thickness)

# Save the marked image (replace 'marked_image.jpg' with your desired filename)
cv2.imwrite('marked_image.jpg', resized_image)

# Display the marked image
cv2.imshow('Marked Image', resized_image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
