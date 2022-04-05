import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec


_GRAY = (250, 250, 250)


def mesh(image):

    image = np.array(Image.fromarray(image).resize((450, 500)))
    mp_face_mesh = mp.solutions.face_mesh

    # Load drawing_utils and drawing_styles
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = DrawingSpec(color=_GRAY, thickness=1)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5
    )

    # Run MediaPipe Face Mesh.
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    results = face_mesh.process(image)

    # Draw face landmarks of each face.
    if not results.multi_face_landmarks:
        return image
    else:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
            )

        return image
