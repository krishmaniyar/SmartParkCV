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
    (60, 5, 140, 50),
    (60, 60, 140, 55),
    (60, 120, 140, 55),
    (60, 180, 140, 55),
    (60, 240, 140, 55),
    (60, 300, 140, 55),
    (60, 360, 140, 55),

    # RIGHT COLUMN
    (220, 5, 140, 50),
    (220, 60, 140, 55),
    (220, 120, 140, 55),
    (220, 180, 140, 55),
    (220, 240, 140, 55),
    (220, 300, 140, 55),
    (220, 360, 140, 55),
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

        if pred == 0: 
            color = (0, 255, 0)
            text_color = (0, 0, 0)
            text = 'EMPTY'
        else:
            color = (0, 0, 255)
            text_color = (255, 255, 255)
            text = 'FULL'

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label_height = 20

        # Filled rectangle for text background
        cv2.rectangle(
            frame,
            (x, y),
            (x + 60, y + label_height),
            color,
            -1
        )

        # Put text inside the filled area
        cv2.putText(
            frame,
            text,
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )

    out.write(frame)   
    cv2.imshow('Parking Lot Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()