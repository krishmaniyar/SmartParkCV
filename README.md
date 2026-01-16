# 🚗 SmartParkCV - Parking Slot Occupancy Detection

A computer vision project that detects whether individual parking slots are **EMPTY** or **OCCUPIED** using a **Support Vector Machine (SVM)** classifier and **OpenCV**. Each parking slot is analyzed independently and highlighted with **green (empty)** or **red (occupied)** bounding boxes in real-time video processing.

---

## 🎥 Demo

![Parking Slot Detection Demo](./files/parking-output.gif)

> 🟢 **Green box** → EMPTY slot  
> 🔴 **Red box** → OCCUPIED slot

---

## 📌 Project Overview

This project implements an intelligent parking lot monitoring system that:

- **Problem**: Automatically detect parking slot availability from video footage
- **Approach**: 
  - Train an SVM classifier on cropped parking slot images (empty vs. occupied)
  - Divide video frames into fixed parking-slot regions of interest (ROIs)
  - Classify each ROI independently in real-time
- **Output**: Real-time visualization, saved video output, and GIF generation

---

## 🧠 Model Details

- **Algorithm**: Support Vector Machine (SVC) with RBF kernel
- **Input Features**:
  - RGB image resized to **15 × 15 pixels**
  - Flattened pixel vector (675 features)
- **Classes**:
  - `0` → Empty slot
  - `1` → Occupied slot
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
  - `C`: [1, 10, 100, 1000]
  - `gamma`: [0.01, 0.001, 0.0001]

---

## 📂 Project Structure

```
SmartParkCV/
├── clf-data/              # Training dataset
│   ├── empty/            # Empty parking slot images (3045 images)
│   └── not_empty/        # Occupied parking slot images (3045 images)
├── files/                # Video files and outputs
│   ├── parking_1920_1080.mp4
│   ├── parking_crop.mp4
│   ├── parking_output.mp4
│   └── parking-output.gif
├── main.py               # Model training script
├── detection.py          # Video detection and processing script
├── model.p              # Trained SVM model (pickle file)
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes core ML libraries. You may also need:
```bash
pip install opencv-python imageio imageio-ffmpeg
```

Or install all at once:
```bash
pip install numpy opencv-python scikit-image scikit-learn imageio imageio-ffmpeg
```

---

## 🏋️ Training the Model

Train the SVM classifier on the parking slot dataset:

```bash
python main.py
```

**What it does**:
1. Loads images from `clf-data/empty/` and `clf-data/not_empty/` directories
2. Resizes each image to 15×15 pixels and flattens to a feature vector
3. Splits data into 80% training and 20% testing sets
4. Performs GridSearchCV to find optimal hyperparameters
5. Trains the best model and saves it as `model.p`
6. Prints the test accuracy score

**Expected Output**:
```
XX% of samples were correctly classified
```

---

## 🎥 Running Parking Slot Detection

Process a video to detect parking slot occupancy:

```bash
python detection.py
```

**What it does**:
1. Loads the trained model from `model.p`
2. Opens the video file (`./files/parking_crop.mp4`)
3. For each frame:
   - Extracts 12 predefined parking slot ROIs
   - Resizes each ROI to 15×15 and classifies it
   - Draws green boxes for empty slots and red boxes for occupied slots
4. Displays real-time visualization
5. Saves the output video to `./files/parking_output.mp4`

**Controls**:
- Press `q` to quit the video display

---

## 🧩 Parking Slot Configuration

The parking slots are manually defined as fixed ROIs (Regions of Interest) in `detection.py`. The current configuration includes:

- **12 parking slots** arranged in 2 columns
- **Left column**: 6 slots
- **Right column**: 6 slots

Each slot is defined as `(x, y, width, height)` coordinates. To modify the parking layout, edit the `parking_slots` list in `detection.py`.

---

## 🔧 Customization

### Using Your Own Video

1. Replace `./files/parking_crop.mp4` with your video file path in `detection.py`
2. Adjust the `parking_slots` coordinates to match your parking lot layout
3. Ensure the video frame dimensions match your ROI coordinates

### Retraining with New Data

1. Add your training images to `clf-data/empty/` and `clf-data/not_empty/`
2. Run `python main.py` to retrain the model
3. The new model will be saved as `model.p`

---

## 📊 Dataset

The project includes a dataset of:
- **3,045 empty parking slot images**
- **3,045 occupied parking slot images**

Total: **6,090 training images**

---

## 🛠️ Technologies Used

- **Python 3**
- **OpenCV** - Video processing and image manipulation
- **scikit-learn** - SVM classifier and model evaluation
- **scikit-image** - Image preprocessing and resizing
- **NumPy** - Numerical operations
- **Pickle** - Model serialization

---

## 👨‍💻 Author

**Krish Maniyar**  
Computer Science Engineering Student

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📝 Notes

- The model is trained on specific parking slot images and may need retraining for different parking lots
- ROI coordinates are hardcoded and should match the training data distribution
- For best results, ensure consistent lighting and camera angle in your video footage

---

## 🚀 Future Improvements

- [ ] Automatic ROI detection using object detection
- [ ] Real-time webcam support
- [ ] REST API for integration
- [ ] Database logging of parking availability
- [ ] Mobile app integration
- [ ] Deep learning model for improved accuracy
