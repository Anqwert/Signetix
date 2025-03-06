import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector (detects a maximum of 1 hand)
detector = HandDetector(maxHands=1)

# Offset for bounding box around hand
offset = 20  

# Fixed image size for saving hand images
imgSize = 300  

# Folder to save captured hand gesture images
folder = "Data/Good Luck"
counter = 0  # Counter for saved images

while True:
    success, img = cap.read()  # Capture frame from webcam
    hands, img = detector.findHands(img)  # Detect hands in the frame
    
    if hands:
        hand = hands[0]  # Get first detected hand
        x, y, w, h = hand['bbox']  # Get bounding box of hand
        
        # Create a white canvas (300x300) for resized hand images
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Adjust bounding box with offset, ensuring it stays within image bounds
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        y2 = min(y + h + offset, img.shape[0])

        # Crop the hand region from the image
        imgCrop = img[y1:y2, x1:x2]

        # Ensure the cropped image is not empty before processing
        if imgCrop.size > 0:
            # Resize the cropped image while maintaining aspect ratio
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)

                # Ensure resized image fits within the white canvas
                imgWhite[:, wGap:wCal + wGap] = imgResize[:, 0:imgWhite.shape[1] - wGap]

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)

                # Ensure resized image fits within the white canvas
                imgWhite[hGap:hCal + hGap, :] = imgResize[0:imgWhite.shape[0] - hGap, :]

            # Display cropped and processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
            
    # Show the main camera feed
    cv2.imshow("Image", img)

    # Capture and save the processed image when 's' is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)

    # Exit the loop when 'q' is pressed
    elif key & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
