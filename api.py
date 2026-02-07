# =========================
# FastAPI ML Exercise Session API (FINAL – real metrics, no dummy values)
# =========================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import json
import random
from typing import Dict, Any, Optional

# ---- Import YOUR project modules from src ----
from src.pose_tracker import PoseTracker
from src.counters import BicepCurlCounter, SquatCounter
from src.utils import calculate_angle
import mediapipe as mp

app = FastAPI()

# =========================
# Initialize ML Components
# =========================

pose_tracker = PoseTracker()
mp_pose = mp.solutions.pose

# =========================
# Helper scoring functions (taken from your final OpenCV code)
# =========================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_from_error(err: float, tol: float) -> float:
    return clamp01(1.0 - (err / tol))


def alignment_score(exercise: str, lm) -> float:
    def pt(e):
        p = lm[e.value]
        return (p.x, p.y)

    LS = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    RS = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    LH = pt(mp_pose.PoseLandmark.LEFT_HIP)
    RH = pt(mp_pose.PoseLandmark.RIGHT_HIP)

    mid_sh = ((LS[0] + RS[0]) / 2, (LS[1] + RS[1]) / 2)
    mid_hip = ((LH[0] + RH[0]) / 2, (LH[1] + RH[1]) / 2)

    down = (mid_hip[0], mid_hip[1] + 0.3)
    torso_angle = calculate_angle(mid_sh, mid_hip, down)
    torso_err = abs(180 - torso_angle)
    torso_score = score_from_error(torso_err, tol=25)

    if exercise == "curl":
        LE = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
        LW = pt(mp_pose.PoseLandmark.LEFT_WRIST)

        elbow_hip_dx = abs(LE[0] - LH[0])
        elbow_close_score = score_from_error(elbow_hip_dx, tol=0.12)

        wrist_elbow_dx = abs(LW[0] - LE[0])
        wrist_stack_score = score_from_error(wrist_elbow_dx, tol=0.10)

        s = 0.45 * torso_score + 0.35 * elbow_close_score + 0.20 * wrist_stack_score
        return 100 * s

    if exercise == "squat":
        LK = pt(mp_pose.PoseLandmark.LEFT_KNEE)
        RK = pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        LA = pt(mp_pose.PoseLandmark.LEFT_ANKLE)
        RA = pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

        left_track = score_from_error(abs(LK[0] - LA[0]), tol=0.10)
        right_track = score_from_error(abs(RK[0] - RA[0]), tol=0.10)
        knee_track = (left_track + right_track) / 2

        mid_ank = ((LA[0] + RA[0]) / 2, (LA[1] + RA[1]) / 2)
        hip_center_err = abs(mid_hip[0] - mid_ank[0])
        hip_center = score_from_error(hip_center_err, tol=0.12)

        s = 0.45 * knee_track + 0.35 * torso_score + 0.20 * hip_center
        return 100 * s

    return 0.0

# =========================
# Session Tracker
# =========================

class SessionTracker:
    def __init__(self, exercise: str, target_reps: int):
        self.exercise = exercise.lower()
        self.target_reps = target_reps

        if self.exercise in ["curl", "bicep_curl"]:
            self.counter = BicepCurlCounter()
        elif self.exercise == "squat":
            self.counter = SquatCounter()
        else:
            raise ValueError(f"Unsupported exercise: {exercise}")

        self.form_scores: list[float] = []
        self.align_scores: list[float] = []
        self.last_angle: Optional[float] = None

    # ---------- Landmark extraction ----------
    def _get_points(self, lm):
        if self.exercise in ["curl", "bicep_curl"]:
            return (
                (lm[11].x, lm[11].y),
                (lm[13].x, lm[13].y),
                (lm[15].x, lm[15].y),
                lm[13].visibility,
            )
        else:
            return (
                (lm[23].x, lm[23].y),
                (lm[25].x, lm[25].y),
                (lm[27].x, lm[27].y),
                lm[25].visibility,
            )

    # ---------- Frame update ----------
    def update(self, frame: np.ndarray):
        if self.counter.state.counter >= self.target_reps:
            return

        results, _ = pose_tracker.process_bgr(frame)
        if not results.pose_landmarks:
            return

        lm = results.pose_landmarks.landmark

        # Alignment score (REAL)
        align_pct = alignment_score(self.exercise, lm)
        self.align_scores.append(align_pct)

        # Rep counter
        info = self.counter.update(*self._get_points(lm))
        angle = info["angle"]

        # Form confidence (REAL from your OpenCV logic)
        if self.exercise == "curl":
            good_range = 20 <= angle <= 160
        else:
            good_range = 80 <= angle <= 180

        score = 70 if good_range else 0

        if self.last_angle is not None:
            angle_change = abs(angle - self.last_angle)
            score += 20 if angle_change < 15 else 5

        self.last_angle = angle

        if info["reliable"]:
            score += 10

        self.form_scores.append(min(100, score))

    # ---------- Final report ----------
    def generate_report(self) -> Dict[str, Any]:
        completed_reps = min(self.counter.state.counter, self.target_reps)
        avg_form = sum(self.form_scores) / max(1, len(self.form_scores))
        avg_align = sum(self.align_scores) / max(1, len(self.align_scores))

        suggestions_map = {
            "curl": [
                "Keep your elbow close to your body (don’t let it drift forward).",
                "Avoid swinging your torso — stay upright.",
                "Control the lowering phase (don’t drop quickly).",
                "Keep wrist neutral (in line with forearm).",
            ],
            "squat": [
                "Keep chest up and spine neutral (avoid rounding).",
                "Knees track outward (avoid caving inward).",
                "Keep heels down and weight mid-foot.",
                "Go as deep as comfortable with control.",
            ],
        }

        suggestions = random.sample(
            suggestions_map.get(self.exercise, []),
            k=min(2, len(suggestions_map.get(self.exercise, []))),
        )

        return {
            "exercise": self.exercise,
            "target_reps": self.target_reps,
            "completed_reps": completed_reps,
            "form_confidence": round(avg_form, 1),
            "alignment": round(avg_align, 1),
            "suggestions": suggestions,
        }

# =========================
# WebSocket Endpoint
# =========================

@app.websocket("/ws/session")
async def session_ws(websocket: WebSocket):
    await websocket.accept()

    session: Optional[SessionTracker] = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"]:
                data = json.loads(message["text"])

                if data.get("type") == "start":
                    session = SessionTracker(
                        exercise=data["exercise"],
                        target_reps=data["target_reps"],
                    )

                elif data.get("type") == "end" and session:
                    report = session.generate_report()
                    await websocket.send_text(json.dumps({"type": "final_report", "report": report}))
                    break

            if "bytes" in message and session:
                np_frame = np.frombuffer(message["bytes"], dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                if frame is not None:
                    session.update(frame)

                    if session.counter.state.counter >= session.target_reps:
                        report = session.generate_report()
                        await websocket.send_text(json.dumps({
                            "type": "final_report",
                            "report": report,
                        }))
                        break

    except WebSocketDisconnect:
        print("Client disconnected")

# =========================
# Health Check
# =========================

@app.get("/")
def root():
    return {"status": "Exercise ML API running with REAL scoring"}


