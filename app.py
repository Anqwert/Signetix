import os
import cv2
import numpy as np
import math
import streamlit as st
from gtts import gTTS
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx3  # Import pyttsx3 for offline TTS (may not work on Android)
import time

# Function to handle speech synthesis using gTTS (works on Android as well)
def speak_text_tts(text, rate=150, volume=1, pitch=100):
    # Use gTTS instead of pyttsx3 on Android
    tts = gTTS(text=text, lang='en', slow=False)
    # Save to a temporary file
    tts.save("output.mp3")
    return "output.mp3"  # Return the path to the audio file

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Load labels dynamically from the 'labels.txt' file
labels = {}
with open("model/labels.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            parts = line.split(" ", 1)
            if len(parts) == 2:
                index, label = parts
                labels[int(index)] = label

# Streamlit settings
st.title("Signetix - Hand Sign Recognition")
st.markdown('<p class="description">Identify gestures in real-time using your hand!</p>', unsafe_allow_html=True)

frame_placeholder = st.empty()

# Initialize session state for gesture history
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []

# Initialize session state for voice feedback status
if 'is_voice_feedback_enabled' not in st.session_state:
    st.session_state.is_voice_feedback_enabled = True  # Default: Voice feedback is enabled

# Add a mode toggle switch in the sidebar
mode_toggle = st.sidebar.radio("Select Mode", ["Camera Mode", "Image Upload Mode"])

# Voice feedback toggle
voice_feedback_toggle = st.sidebar.checkbox("Enable Voice Feedback", value=True)
st.session_state.is_voice_feedback_enabled = voice_feedback_toggle

# Gradient shadow effect toggle
gradient_shadow_toggle = st.sidebar.checkbox("Enable Gradient Shadow Effect", value=False)

# Initialize the audio_file variable as None before the if block
audio_file = None

# Camera Mode - Live Feed
if mode_toggle == "Camera Mode":
    # Live Camera Mode - Same as your original logic
    video_input = st.camera_input("Take a picture")

    if video_input:
        # Process the video feed here (same as your existing logic)
        progress_bar = st.progress(0)
        img = cv2.imdecode(np.frombuffer(video_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        progress_bar.progress(10)

        # Find hands and landmarks
        hands, img = detector.findHands(img)
        progress_bar.progress(30)

        if hands:
            hand = hands[0]
            landmarks = hand['lmList']  # List of the 21 keypoints of the hand
            x, y, w, h = hand['bbox']

            # Add the offset around the bounding box
            x1 = max(x - 20, 0)  # Ensure x1 is not negative
            y1 = max(y - 20, 0)  # Ensure y1 is not negative
            x2 = min(x + w + 20, img.shape[1])  # Ensure x2 does not exceed image width
            y2 = min(y + h + 20, img.shape[0])  # Ensure y2 does not exceed image height

            # Crop the image with the adjusted coordinates
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:  # Ensure the cropped image is not empty
                # Resize imgCrop to fit into a 300x300 canvas
                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = 300 / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, 300))
                    wGap = math.ceil((300 - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = 300 / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (300, hCal))
                    hGap = math.ceil((300 - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Update progress bar to indicate gesture prediction is happening
                progress_bar.progress(60)

                # Make gesture prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                label_text = labels[index]

                # Update progress bar to indicate gesture prediction is complete
                progress_bar.progress(90)

                # Speak the recognized gesture using gTTS if voice feedback is enabled
                if st.session_state.is_voice_feedback_enabled:
                    audio_file = speak_text_tts(f" {label_text}.", rate=150, volume=1, pitch=120)

                # Update the gesture history (keep the last 5 gestures)
                st.session_state.gesture_history.append(label_text)
                if len(st.session_state.gesture_history) > 5:
                    st.session_state.gesture_history.pop(0)

                # Display the predicted gesture text in Streamlit
                st.write(f"Predicted Gesture: {label_text}")

                # Check if gradient effect is enabled
                if not gradient_shadow_toggle:
                    img = cv2.imdecode(np.frombuffer(video_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)

                else:
                    img = cv2.imdecode(np.frombuffer(video_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    start_color = (255, 99, 71)  # Tomato color (for example)
                    end_color = (255, 165, 0)  # Orange color

                    # Draw the gradient-like color using cv2
                    overlay = img.copy()
                    for i in range(50):
                        alpha = i / 50
                        blended = cv2.addWeighted(overlay, 1 - alpha, np.full_like(overlay, end_color), alpha, 0)
                        cv2.rectangle(blended, (x - 20, y - 20 - 50),
                                      (x - 20 + 90, y - 20 - 50 + 50), (int(255 * alpha), int(99 * alpha), 71), -1)

                # Add the bounding box with elegant color and thickness
                    cv2.rectangle(img, (x - 20, y - 20),
                              (x + w + 20, y + h + 20), (255, 165, 0), 4)  # Light orange color with thickness 4

                # Enhanced label with shadow and custom font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_size = 1.2
                    font_thickness = 3
                    label_color = (255, 255, 255)  # White color for the label text
                    shadow_color = (0, 0, 0)  # Black shadow for text

                # Position the label inside the rectangle
                    label_x = x
                    label_y = y - 20 - 50 + 30

                # Create shadow for text
                    cv2.putText(img, label_text, (label_x + 5, label_y + 5), font, font_size, shadow_color, font_thickness + 1)

                # Add the label
                    cv2.putText(img, label_text, (label_x, label_y), font, font_size, label_color, font_thickness)

            # Update the displayed frame (without landmarks or bounding box)
            frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Video Feed", use_column_width=True)

            # Complete the progress bar
            progress_bar.progress(100)

        else:
            # Create an image with a "No Hand Detected" message, centered in the middle
            no_hand_frame = np.ones((300, 300, 3), np.uint8) * 255  # White background

            # Text properties
            text = "No Hand Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # Calculate the position for centered text
            text_x = (no_hand_frame.shape[1] - text_size[0]) // 2
            text_y = (no_hand_frame.shape[0] + text_size[1]) // 2

            # Add the centered text to the frame
            cv2.putText(no_hand_frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # Update the frame placeholder with the centered "No Hand Detected" message
            frame_placeholder.image(cv2.cvtColor(no_hand_frame, cv2.COLOR_BGR2RGB), caption="No Hand Detected", use_column_width=True)

            # Show error message in Streamlit app
            st.error("No hand detected. Please try again.")  # Show error message in the Streamlit app

# Image Upload Mode
if mode_toggle == "Image Upload Mode":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Read and process the uploaded image
        progress_bar = st.progress(0)
        img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        progress_bar.progress(10)

        # Same logic as camera mode for processing the image
        hands, img = detector.findHands(img)
        progress_bar.progress(30)

        if hands:
            hand = hands[0]
            landmarks = hand['lmList']  # List of the 21 keypoints of the hand
            x, y, w, h = hand['bbox']

            # Add the offset around the bounding box
            x1 = max(x - 20, 0)  # Ensure x1 is not negative
            y1 = max(y - 20, 0)  # Ensure y1 is not negative
            x2 = min(x + w + 20, img.shape[1])  # Ensure x2 does not exceed image width
            y2 = min(y + h + 20, img.shape[0])  # Ensure y2 does not exceed image height

            # Crop the image with the adjusted coordinates
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:  # Ensure the cropped image is not empty
                # Resize imgCrop to fit into a 300x300 canvas
                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = 300 / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, 300))
                    wGap = math.ceil((300 - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = 300 / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (300, hCal))
                    hGap = math.ceil((300 - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Update progress bar to indicate gesture prediction is happening
                progress_bar.progress(60)

                # Make gesture prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                label_text = labels[index]

                # Update progress bar to indicate gesture prediction is complete
                progress_bar.progress(90)

                # Speak the recognized gesture using gTTS if voice feedback is enabled
                if st.session_state.is_voice_feedback_enabled:
                    audio_file = speak_text_tts(f" {label_text}.", rate=150, volume=1, pitch=120)

                # Update the gesture history (keep the last 5 gestures)
                st.session_state.gesture_history.append(label_text)
                if len(st.session_state.gesture_history) > 5:
                    st.session_state.gesture_history.pop(0)

                # Display the predicted gesture text in Streamlit
                st.write(f"Predicted Gesture: {label_text}")

                # Check if gradient effect is enabled
                if not gradient_shadow_toggle:
                    img = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)

                else:
                    img = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    start_color = (255, 99, 71)  # Tomato color (for example)
                    end_color = (255, 165, 0)  # Orange color

                    # Draw the gradient-like color using cv2
                    overlay = img.copy()
                    for i in range(50):
                        alpha = i / 50
                        blended = cv2.addWeighted(overlay, 1 - alpha, np.full_like(overlay, end_color), alpha, 0)
                        cv2.rectangle(blended, (x - 20, y - 20 - 50),
                                      (x - 20 + 90, y - 20 - 50 + 50), (int(255 * alpha), int(99 * alpha), 71), -1)

                # Add the bounding box with elegant color and thickness
                    cv2.rectangle(img, (x - 20, y - 20),
                              (x + w + 20, y + h + 20), (255, 165, 0), 4)  # Light orange color with thickness 4

                # Enhanced label with shadow and custom font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_size = 1.2
                    font_thickness = 3
                    label_color = (255, 255, 255)  # White color for the label text
                    shadow_color = (0, 0, 0)  # Black shadow for text

                # Position the label inside the rectangle
                    label_x = x
                    label_y = y - 20 - 50 + 30

                # Create shadow for text
                    cv2.putText(img, label_text, (label_x + 5, label_y + 5), font, font_size, shadow_color, font_thickness + 1)

                # Add the label
                    cv2.putText(img, label_text, (label_x, label_y), font, font_size, label_color, font_thickness)

            # Update the displayed frame (without landmarks or bounding box)
            frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            # Complete the progress bar
            progress_bar.progress(100)

        else:
            st.error("No hand detected in the image.")

# Display gesture history on the side
if st.session_state.gesture_history:
    st.sidebar.subheader("Gesture History")
    for i, gesture in enumerate(reversed(st.session_state.gesture_history)):
        st.sidebar.write(f"{i+1}. {gesture}")

# Display voice feedback status in the sidebar
if st.session_state.is_voice_feedback_enabled:
    st.sidebar.write("Voice Feedback is Enabled")
else:
    st.sidebar.write("Voice Feedback is Disabled")

# Audio button: Let the user click to play the audio if voice feedback is enabled
if audio_file and st.session_state.is_voice_feedback_enabled:
    st.audio(audio_file, format="audio/mp3")
