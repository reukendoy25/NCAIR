import dlib
import cv2
import os

# Path to the shape predictor model
predictor_path = 'C:\\NCAIR\\Asim Sir work\\VS code\\Setero_Vision\\shape_predictor_68_face_landmarks.dat'

# Verify the file path
if not os.path.isfile(predictor_path):
    raise FileNotFoundError(f"The file {predictor_path} does not exist. Please check the path.")

# Print the file path for verification
print(f"Using shape predictor model: {predictor_path}")

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Load the shape predictor model
predictor = dlib.shape_predictor(predictor_path)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

# Set video frame width and height (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit the video")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over each detected face
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Loop over each landmark and draw it on the frame
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    # Display the resulting frame
    cv2.imshow('Live Landmarks', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
