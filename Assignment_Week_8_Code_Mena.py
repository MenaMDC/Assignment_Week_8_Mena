import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time

# Create sidebar for settings
st.sidebar.title("Detection Settings")
face_conf_threshold = st.sidebar.slider(
    "Face Detection Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    step=.01, 
    value=0.3,
    help="Adjust this to filter out low-confidence detections"
)

min_neighbors = st.sidebar.slider(
    "Eye Detection Sensitivity", 
    min_value=1, 
    max_value=15,
    value=3,
    help="Lower values detect more eyes but may include false positives"
)

# Create application title and file uploader widget.
st.title("Face and Eye Detection App")
img_file_buffer = st.file_uploader("Choose an image file - recommended : at least 200x200 pixels", type=['jpg', 'jpeg', 'png'])


# Function for detecting faces in an image.
def detectFaceAndEyes(net, eye_cascade, eye_glass_cascade, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections, eye_cascade, eye_glass_cascade


# Function for annotating the image with bounding boxes for faces and eyes.
def process_detections(frame, detections, eye_cascade, eye_glass_cascade, conf_threshold=0.5, min_neighbors=5):
    bboxes = []
    confidences = []  # List to store confidence values
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    
    # Convert to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Get face coordinates
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w-1, x2), min(frame_h-1, y2)
            
            bboxes.append([x1, y1, x2, y2])
            confidences.append(confidence)  # Store confidence value
            
            # Extract the face ROI
            roi_gray = gray[y1:y2, x1:x2]
            roi_color = frame[y1:y2, x1:x2]
            
            # Get face region dimensions
            face_height = y2 - y1
            face_width = x2 - x1
            
            # Apply preprocessing steps
            # 1. Denoise
            roi_gray = cv2.fastNlMeansDenoising(roi_gray)
            # 2. Apply CLAHE
            roi_gray_eq = clahe.apply(roi_gray)
            # 3. Gaussian blur to reduce noise
            roi_gray_eq = cv2.GaussianBlur(roi_gray_eq, (3,3), 0)
            
            # Try both eye detectors with more lenient parameters
            eyes1 = eye_cascade.detectMultiScale(
                roi_gray_eq,
                scaleFactor=1.03,  # More gradual scaling
                minNeighbors=min_neighbors,
                minSize=(15, 15)
            )
            
            eyes2 = eye_glass_cascade.detectMultiScale(
                roi_gray_eq,
                scaleFactor=1.03,  # More gradual scaling
                minNeighbors=min_neighbors,
                minSize=(15, 15)
            )
            
            # Combine detected eyes from both classifiers
            eyes = np.vstack([eyes1, eyes2]) if len(eyes1) > 0 and len(eyes2) > 0 else \
                  eyes1 if len(eyes1) > 0 else eyes2
            
            # Filter out detections that are likely to be mouths or false positives
            filtered_eyes = []
            if len(eyes) > 0:
                for eye in eyes:
                    ex, ey, ew, eh = eye
                    # Convert to relative positions in face (0-1 range)
                    rel_y = ey / face_height
                    rel_x = ex / face_width
                    
                    # Eyes are typically in the upper half of the face
                    # and within reasonable horizontal bounds
                    if (0.15 <= rel_y <= 0.55 and  # Vertical position
                        0.1 <= rel_x <= 0.9):      # Horizontal position
                        filtered_eyes.append(eye)
            
            # Draw face box
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            
            # Draw green box for face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
            
            # Draw blue circles for detected eyes
            for (ex, ey, ew, eh) in filtered_eyes:
                # Convert eye coordinates to original image coordinates
                eye_center_x = x1 + ex + ew//2
                eye_center_y = y1 + ey + eh//2
                eye_radius = max(ew, eh) // 2
                
                # Draw blue circle around each eye
                cv2.circle(frame, (eye_center_x, eye_center_y), 
                          eye_radius, (255, 0, 0), 2)
    
    return frame, bboxes, confidences


# Function to load the models.
@st.cache_resource()
def load_model():
    # Load face detection model
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    
    # Load both eye detection models
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_glass_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    return net, eye_cascade, eye_glass_cascade


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


net, eye_cascade, eye_glass_cascade = load_model()

if img_file_buffer is not None:
    # Start timing the processing
    start_time = time.time()
    
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Call the detection models
    detections, eye_cascade, eye_glass_cascade = detectFaceAndEyes(net, eye_cascade, eye_glass_cascade, image)

    # Process the detections
    out_image, bboxes, confidences = process_detections(image, detections, eye_cascade, eye_glass_cascade,
                                                      conf_threshold=face_conf_threshold,
                                                      min_neighbors=min_neighbors)

    # Display Detected faces and eyes
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Create a clean metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Faces Detected", len(bboxes))
    with col2:
        st.metric("Processing Time", f"{(time.time() - start_time):.2f}s")
    with col3:
        if confidences:
            st.metric("Avg Confidence", f"{(sum(confidences)/len(confidences)):.2%}")

    # Convert opencv image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "face_eye_detection.jpg", 'Download Output Image'),
                unsafe_allow_html=True)