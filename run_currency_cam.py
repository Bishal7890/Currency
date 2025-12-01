import cv2
import numpy as np
import tensorflow as tf

# ================= CONFIGURATION =================
# 1. Path to your downloaded model file
# Ensure this file is in the same folder as this script
MODEL_PATH = 'currency_efficientnet_final.keras' 

# 2. Image size (EfficientNet standard is 224)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 3. Class Names (Based on alphabetical folder order)
# Try this list first. If it's wrong, just swap the names around.
class_names = [
    '10 Rupee (New)', 
    '100 Rupee', 
    '20 Rupee', 
    '200 Rupee', 
    '2000 Rupee', 
    '50 Rupee', 
    '10 Rupee (Old)'  # This corresponds to Index 6
]
# =================================================

def main():
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model.\n{e}")
        print("Make sure 'currency_efficientnet_final.keras' is in this folder.")
        return

    # Initialize Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting video stream...")
    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- PREPROCESSING ---
        # Resize to model input size
        input_img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        
        # Convert BGR (OpenCV) to RGB (TensorFlow)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # Expand dims to make it a batch of 1: (1, 224, 224, 3)
        input_data = np.expand_dims(input_img, axis=0)

        # --- PREDICTION ---
        predictions = model.predict(input_data, verbose=0)
        
        # Apply Softmax to get probabilities
        score = tf.nn.softmax(predictions[0])
        
        # Get the highest confidence class
        class_idx = np.argmax(score)
        confidence = 100 * np.max(score)
        
        # Safe lookup for class name
        if class_idx < len(class_names):
            label = class_names[class_idx]
        else:
            label = f"Unknown Class {class_idx}"

        # --- VISUALIZATION ---
        # Display text on screen
        text = f"{label}: {confidence:.1f}%"
        
        # Color logic: Green if confident (>70%), Red if unsure
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        
        # Draw rectangle background for text (for better visibility)
        cv2.rectangle(frame, (10, 10), (350, 50), (0,0,0), -1) 
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show the frame
        cv2.imshow('Currency Detector', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()