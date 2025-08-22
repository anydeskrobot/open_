import cv2
from ultralytics import YOLO

def detect_and_display(image_path, model_path="best.pt", conf_threshold=0.5):
    """
    Detect objects in an image using YOLO and display the results
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to YOLO model weights (.pt file)
        conf_threshold (float): Confidence threshold (0-1)
    """
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Perform detection
    results = model.predict(image, conf=conf_threshold)
    
    # Extract and print detection information
    print("\nDetection Results:")
    print("-----------------")
    for i, detection in enumerate(results[0].boxes.data.tolist()):
        x1, y1, x2, y2, conf, class_id = detection
        class_name = results[0].names[int(class_id)]
        print(f"{i+1}. {class_name} (Confidence: {conf:.2f}) at coordinates ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    
    # Get the annotated image with bounding boxes
    annotated_image = results[0].plot()
    
    # Display the original and detected images side by side
    combined = cv2.hconcat([image, annotated_image])
    
    # Resize if too large for screen
    height, width = combined.shape[:2]
    if width > 1920:  # If wider than 1920 pixels
        scale = 1920 / width
        combined = cv2.resize(combined, (int(width*scale), int(height*scale)))
    
    cv2.imshow("Original (Left) vs Detected (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    detect_and_display(
        image_path="/home/thunder/adhesh/opendroid/opendroid/sree/00359.jpg",  # Replace with your image path
        model_path="best.pt",  # Can use yolov8s.pt, yolov8m.pt, etc.
        conf_threshold=0.5
    )
