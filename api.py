# =========================
# FastAPI Exercise Verification API (FINAL – frontend compatible)
# =========================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import json
import random
from typing import Dict, Any, Optional, List

# ---- Project imports ----
from src.pose_tracker import PoseTracker
from src.counters import BicepCurlCounter, SquatCounter
from src.utils import calculate_angle
import mediapipe as mp

app = FastAPI()

# =========================
# ML Initialization
# =========================

pose_tracker = PoseTracker()
mp_pose = mp.solutions.pose

# =========================
# Scoring helper
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

        elbow_dx = abs(LE[0] - LH[0])
        wrist_dx = abs(LW[0] - LE[0])

        s = (
            0.45 * torso_score
            + 0.35 * score_from_error(elbow_dx, 0.12)
            + 0.20 * score_from_error(wrist_dx, 0.10)
        )
        return round(100 * s, 1)

    if exercise == "squat":
        LK = pt(mp_pose.PoseLandmark.LEFT_KNEE)
        RK = pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        LA = pt(mp_pose.PoseLandmark.LEFT_ANKLE)
        RA = pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

        knee_track = (
            score_from_error(abs(LK[0] - LA[0]), 0.10)
            + score_from_error(abs(RK[0] - RA[0]), 0.10)
        ) / 2

        mid_ank = ((LA[0] + RA[0]) / 2, (LA[1] + RA[1]) / 2)
        hip_center = score_from_error(abs(mid_hip[0] - mid_ank[0]), 0.12)

        s = 0.45 * knee_track + 0.35 * torso_score + 0.20 * hip_center
        return round(100 * s, 1)

    return 0.0


# =========================
# Session Tracker
# =========================

class SessionTracker:
    def __init__(self, exercise: str, target_reps: int):
        self.exercise = exercise
        self.target_reps = target_reps

        if exercise == "curl":
            self.counter = BicepCurlCounter()
        elif exercise == "squat":
            self.counter = SquatCounter()
        else:
            raise ValueError("Unsupported exercise")

        self.form_scores: List[float] = []
        self.align_scores: List[float] = []
        self.last_angle: Optional[float] = None

    def _points(self, lm):
        if self.exercise == "curl":
            return (
                (lm[11].x, lm[11].y),
                (lm[13].x, lm[13].y),
                (lm[15].x, lm[15].y),
                lm[13].visibility,
            )
        return (
            (lm[23].x, lm[23].y),
            (lm[25].x, lm[25].y),
            (lm[27].x, lm[27].y),
            lm[25].visibility,
        )

    def update(self, frame: np.ndarray):
        results, _ = pose_tracker.process_bgr(frame)
        if not results.pose_landmarks:
            return

        lm = results.pose_landmarks.landmark

        # Visibility gate (CRITICAL)
        vis_idx = 13 if self.exercise == "curl" else 25
        if lm[vis_idx].visibility < 0.6:
            return

        self.align_scores.append(alignment_score(self.exercise, lm))

        info = self.counter.update(*self._points(lm))
        angle = info["angle"]

        # Form score
        good_range = (
            20 <= angle <= 160 if self.exercise == "curl"
            else 80 <= angle <= 180
        )

        score = 70 if good_range else 0
        if self.last_angle is not None:
            score += 20 if abs(angle - self.last_angle) < 15 else 5
        if info["reliable"]:
            score += 10

        self.last_angle = angle
        self.form_scores.append(min(100, score))

    def report(self) -> Dict[str, Any]:
        avg_form = sum(self.form_scores) / max(1, len(self.form_scores))
        avg_align = sum(self.align_scores) / max(1, len(self.align_scores))

        suggestions = {
            "curl": [
                "Keep your elbow close to your body.",
                "Avoid swinging your torso.",
                "Control the lowering phase.",
            ],
            "squat": [
                "Keep chest up and spine neutral.",
                "Push knees outward.",
                "Keep heels grounded.",
            ],
        }

        return {
            "exercise": self.exercise,
            "target_reps": self.target_reps,
            "completed_reps": self.counter.state.counter,
            "form_confidence": round(avg_form, 1),
            "alignment": round(avg_align, 1),
            "suggestions": random.sample(
                suggestions[self.exercise],
                k=min(2, len(suggestions[self.exercise])),
            ),
        }


# =========================
# WebSocket Endpoint
# =========================

@app.websocket("/ws/session")
async def ws_session(ws: WebSocket):
    await ws.accept()
    session: Optional[SessionTracker] = None

    try:
        while True:
            msg = await ws.receive()

            # -------- CONTROL (JSON) --------
            if "text" in msg:
                data = json.loads(msg["text"])

                if data["type"] == "start":
                    session = SessionTracker(
                        exercise=data["exercise"],
                        target_reps=int(data["target_reps"]),
                    )

                elif data["type"] == "end" and session:
                    await ws.send_json({
                        "type": "final_report",
                        "report": session.report(),
                    })
                    return

            # -------- FRAME (JPEG) --------
            if "bytes" in msg and session:
                frame = cv2.imdecode(
                    np.frombuffer(msg["bytes"], np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    continue

                session.update(frame)

                # LIVE UPDATE (frontend expects this)
                await ws.send_json({
                    "type": "live",
                    "reps": session.counter.state.counter,
                    "target_reps": session.target_reps,
                    "form_confidence": round(
                        sum(session.form_scores[-10:]) / max(1, len(session.form_scores[-10:])),
                        1,
                    ),
                    "alignment": round(
                        sum(session.align_scores[-10:]) / max(1, len(session.align_scores[-10:])),
                        1,
                    ),
                    "suggestions": session.report()["suggestions"],
                })

                if session.counter.state.counter >= session.target_reps:
                    await ws.send_json({
                        "type": "final_report",
                        "report": session.report(),
                    })
                    return

    except WebSocketDisconnect:
        print("WebSocket disconnected")


# =========================
# Health
# =========================

@app.get("/")
def health():
    return {"status": "ok"}
