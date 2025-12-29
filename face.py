import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
image_path = input("Enter the path to the image: ")

frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(image_rgb)
if not results:
    print("No faces detected in the image.")
else:
    for face in results:
        x, y, width, height = face['box']
        confidence = face['confidence']
        
        if confidence > 0.90:
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            
            keypoints = face['keypoints']
            cv2.circle(frame, keypoints['left_eye'], 2, (0, 0, 255), 2)
            cv2.circle(frame, keypoints['right_eye'], 2, (0, 0, 255), 2)
            cv2.circle(frame, keypoints['nose'], 2, (0, 0, 255), 2)
            cv2.circle(frame, keypoints['mouth_left'], 2, (0, 0, 255), 2)
            cv2.circle(frame, keypoints['mouth_right'], 2, (0, 0, 255), 2)


cv2.imshow("Face Detection (MTCNN)", frame)
cv2.waitKey(0)  
cv2.destroyAllWindows()