import cv2
import pickle
import numpy as np
from skimage.transform import resize

# Load model
model = pickle.load(open('./model.p', 'rb'))
categories = ['empty', 'not_empty']

# Parking slot ROIs (x, y, w, h)
parking_slots = [
    # LEFT COLUMN
    (30, 30, 170, 60),
    (30, 100, 170, 60),
    (30, 180, 170, 60),
    (30, 240, 170, 60),
    (30, 300, 170, 60),
    (30, 370, 170, 60),

    # RIGHT COLUMN
    (220, 30, 160, 60),
    (220, 100, 160, 60),
    (220, 180, 160, 60),
    (220, 240, 160, 60),
    (220, 300, 160, 60),
    (220, 370, 160, 60),
]


cap = cv2.VideoCapture('./files/parking_crop.mp4')

# Video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = 20

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./files/parking_output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for (x, y, w, h) in parking_slots:
        roi = frame[y:y+h, x:x+w]

        # Convert & resize
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = resize(roi_rgb, (15, 15))
        roi_flat = roi_resized.flatten().reshape(1, -1)

        # Predict
        pred = model.predict(roi_flat)[0]

        if pred == 0:  # empty
            color = (0, 255, 0)
            text = 'EMPTY'
        else:
            color = (0, 0, 255)
            text = 'FULL'

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out.write(frame)   
    cv2.imshow('Parking Lot Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()