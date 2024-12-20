import torch
import torchvision
from torchvision import transforms as T
import cv2

model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the transformation to convert image to tensor
imgtransform = T.ToTensor()

while True:
    # Capture frame-by-frame
    ret, image = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    img = image.copy()

    # Convert image to tensor
    image_tensor = imgtransform(image)

    # Perform object detection using the model
    with torch.no_grad():
        ypred = model([image_tensor])

        # Extract bounding boxes, scores, and labels
        bbox, scores, labels = ypred[0]['boxes'], ypred[0]['scores'], ypred[0]['labels']

        # Filter detections with a confidence score greater than 0.8
        nums = torch.argwhere(scores > 0.80).shape[0]

        # Loop through the detected objects and draw bounding boxes and labels
        for i in range(nums):
            x, y, w, h = bbox[i].numpy().astype('int')
            cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 5)
            classname = labels[i].numpy().astype('int')
            classdetected = classnames[classname - 1]  # Get the class name

            # Display class label on the image using OpenCV's putText
            cv2.putText(img, classdetected, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('Real-Time Object Detection', img)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
