from ultralytics import YOLO
import cv2 # Load OpenCV for image processing

model = YOLO("yolov8n.pt")

image_path = ["sample.jpg", "sample2.png"]  # List of image paths
images = [cv2.imread(path) for path in image_path]
results = model(images)  

for i, result in enumerate(results):
    annotated = result.plot()
    cv2.imshow(f"Result {i}",annotated)


cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the OpenCV windowT
