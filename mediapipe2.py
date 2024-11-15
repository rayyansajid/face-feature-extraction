import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load an image and prepare it for processing
# image_path = r"./face.jpeg"  # Replace with your image path
# image_path = r"./blackMan2.jpeg"  # Replace with your image path
# image_path = r"./ChineseFace.jpeg"  # Replace with your image path
# image_path = r"./blackMan.jpeg"  # Replace with your image path
image_path = r"./blackWoman.jpeg"  # Replace with your image path
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with Face Mesh
results = face_mesh.process(rgb_image)

# Initialize dictionary to store face description
face_description = {}

# Helper function to calculate Euclidean distance between two landmarks
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Extract facial attributes if any faces are detected
if results.multi_face_landmarks:
    # print(f"\t\n\tresults.multi_face_landmarks:\n{results.multi_face_landmarks}\n#######\n\n")
    for face_landmarks in results.multi_face_landmarks:
        # 1. Eye Shape (as per your original code)
        left_eye_outer = face_landmarks.landmark[33]
        left_eye_inner = face_landmarks.landmark[133]
        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        left_eye_width = calculate_distance(left_eye_outer, left_eye_inner)
        left_eye_height = calculate_distance(left_eye_top, left_eye_bottom)

        right_eye_outer = face_landmarks.landmark[362]
        right_eye_inner = face_landmarks.landmark[398]
        right_eye_top = face_landmarks.landmark[386]
        right_eye_bottom = face_landmarks.landmark[374]
        right_eye_width = calculate_distance(right_eye_outer, right_eye_inner)
        right_eye_height = calculate_distance(right_eye_top, right_eye_bottom)

        left_eye_ratio = left_eye_width / left_eye_height
        right_eye_ratio = right_eye_width / right_eye_height
        face_description["eye_shape"] = "Almond-shaped" if left_eye_ratio > 1.8 and right_eye_ratio > 1.8 else "Round"

        # 2. Face Shape (based on jawline and cheekbones)
        chin = face_landmarks.landmark[152]
        left_cheekbone = face_landmarks.landmark[234]
        right_cheekbone = face_landmarks.landmark[454]
        forehead_center = face_landmarks.landmark[10]
        jaw_width = calculate_distance(left_cheekbone, right_cheekbone)
        face_height = calculate_distance(forehead_center, chin)

        if jaw_width / face_height > 0.85:
            face_description["face_shape"] = "Square"
        elif jaw_width / face_height > 0.75:
            face_description["face_shape"] = "Oval"
        else:
            face_description["face_shape"] = "Heart"

        # 3. Mouth Shape (width of mouth vs height)
        left_mouth_corner = face_landmarks.landmark[61]
        right_mouth_corner = face_landmarks.landmark[291]
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_width = calculate_distance(left_mouth_corner, right_mouth_corner)
        mouth_height = calculate_distance(upper_lip, lower_lip)
        face_description["mouth_shape"] = "Wide" if mouth_width / mouth_height > 2.0 else "Full"

        # 4. Improved Skin Tone (sample region and use YCrCb color space)
        cheek_x = int(left_cheekbone.x * image.shape[1])
        cheek_y = int(left_cheekbone.y * image.shape[0])
        cheek_region = image[cheek_y - 10:cheek_y + 10, cheek_x - 10:cheek_x + 10]  # 20x20 pixel region around the cheek

        # Convert the region to YCrCb color space
        cheek_region_ycrcb = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2YCrCb)
        avg_ycrcb = np.mean(cheek_region_ycrcb, axis=(0, 1))

        # Classify based on Cr component (chrominance)
        if avg_ycrcb[1] > 145:
            face_description["skin_tone"] = "Fair"
        elif avg_ycrcb[1] > 130:
            face_description["skin_tone"] = "Medium"
        else:
            face_description["skin_tone"] = "Dark"

# Output the generated face description
for attribute, description in face_description.items():
    print(f"{attribute.capitalize()}: {description}")

# Clean up
face_mesh.close()
