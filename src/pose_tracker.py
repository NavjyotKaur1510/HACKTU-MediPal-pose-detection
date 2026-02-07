import cv2
import mediapipe as mp

class PoseTracker:
    def __init__(self, min_det_conf=0.5, min_track_conf=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )

    def process_bgr(self, frame_bgr):
        """
        Returns (results, frame_bgr_annotated)
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = frame_bgr.copy()

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return results, annotated
