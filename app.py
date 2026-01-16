import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load Model & Setup
model = load_model('best_emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Colors for the bars (BGR format)
colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,255,255), (255,0,255), (100,100,100)]

while True:
    ret, frame = cap.read()
    if not ret: break

    # Create a canvas for the UI (640px wide for video + 300px for sidebar)
    canvas = np.zeros((480, 940, 3), dtype="uint8")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Default prediction data if no face is found
    preds = [0] * 7
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocessing
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Prediction
        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]
        
        # Draw main label on video
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- BUILD THE UI SIDEBAR ---
    for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
        # Calculate bar width based on probability (max 200px)
        w = int(prob * 200)
        
        # Draw the text label
        cv2.putText(canvas, emotion, (10, (i * 50) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # Draw the bar
        cv2.rectangle(canvas, (100, (i * 50) + 33), (100 + w, (i * 50) + 55), colors[i], -1)
        # Draw percentage text
        cv2.putText(canvas, f"{int(prob*100)}%", (110 + w, (i * 50) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Stitch the video frame and the UI canvas together
    canvas[:480, :640] = frame
    
    # Add a Footer
    cv2.rectangle(canvas, (0, 450), (940, 480), (50, 50, 50), -1)
    cv2.putText(canvas, "Press 'Q' to Exit | Emotion Analysis Dashboard", (300, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Emotion AI - Interactive Pro', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()