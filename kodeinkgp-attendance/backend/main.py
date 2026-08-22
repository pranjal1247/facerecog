import os
import json
import base64
import asyncio
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="KodeinKGP HUD Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # wildcard origin + credentials=True is an invalid combo
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.json")
PROFILES_PATH = os.path.join(DATA_DIR, "profiles.json")

print("[Init] Loading FAISS Index and Metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "r") as f:
    id_map = json.load(f)
with open(PROFILES_PATH, "r") as f:
    profiles = json.load(f)

print("[Init] Loading InsightFace Model...")
app_face = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
app_face.prepare(ctx_id=0, det_size=(224, 224))

from insightface.app.common import Face

# Pull the individual sub-models out so we can call detection and
# recognition independently at different frequencies.
det_model = app_face.models['detection']
rec_model = app_face.models['recognition']

RECOGNITION_INTERVAL_S = 0.4   # only re-verify identity this often per track

import time
from collections import deque, Counter

SIM_THRESHOLD = 0.40
HISTORY_LEN = 7          # frames of history per track used for majority vote, higher = smoother but slower to lock on
MATCH_DIST_PX = 180      # max centroid movement (in orig frame px) to match same track, raise if you move around a lot in frame
STALE_AFTER_S = 1.5      # drop a track if not seen for this long


class FaceTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def match(self, bbox):
        """Match a raw detection to an existing track by centroid proximity.
        Returns the track_id (existing or freshly allocated) WITHOUT touching
        identity — that's decided separately based on recognition timing."""
        centroid = self._centroid(bbox)
        best_id, best_dist = None, None

        for tid, t in self.tracks.items():
            dist = ((t["centroid"][0] - centroid[0]) ** 2 +
                    (t["centroid"][1] - centroid[1]) ** 2) ** 0.5
            if dist < MATCH_DIST_PX and (best_dist is None or dist < best_dist):
                best_id, best_dist = tid, dist

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
            self.tracks[best_id] = {
                "history": deque(maxlen=HISTORY_LEN),
                "last_recognized": 0,
                "identity": "UNKNOWN",
            }

        self.tracks[best_id]["centroid"] = centroid
        self.tracks[best_id]["last_seen"] = time.time()
        return best_id

    def needs_recognition(self, track_id):
        t = self.tracks[track_id]
        return (time.time() - t["last_recognized"]) >= RECOGNITION_INTERVAL_S

    def report_identity(self, track_id, raw_identity):
        t = self.tracks[track_id]
        t["history"].append(raw_identity)
        t["last_recognized"] = time.time()
        t["identity"] = Counter(t["history"]).most_common(1)[0][0]

    def current_identity(self, track_id):
        return self.tracks[track_id]["identity"]

    def cleanup(self):
        now = time.time()
        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid].get("last_seen", 0) > STALE_AFTER_S:
                del self.tracks[tid]


def run_inference(frame, orig_w, orig_h, tracker):
    scale_x = orig_w / frame.shape[1]
    scale_y = orig_h / frame.shape[0]

    # Fast path: detection only, every single frame
    bboxes, kpss = det_model.detect(frame, max_num=0, metric='default')
    tracks_out = []

    for i in range(bboxes.shape[0]):
        raw_bbox = bboxes[i, 0:4]
        kps = kpss[i] if kpss is not None else None

        scaled_bbox = [
            int(raw_bbox[0] * scale_x),
            int(raw_bbox[1] * scale_y),
            int(raw_bbox[2] * scale_x),
            int(raw_bbox[3] * scale_y),
        ]

        track_id = tracker.match(scaled_bbox)

        # Slow path: only run embedding + FAISS when this track is "due"
        if tracker.needs_recognition(track_id) and kps is not None:
            face = Face(bbox=raw_bbox, kps=kps, det_score=1.0)
            rec_model.get(frame, face)

            emb_norm = (face.embedding / np.linalg.norm(face.embedding)).astype('float32').reshape(1, -1)
            sims, idxs = index.search(emb_norm, k=1)
            score = float(sims[0][0])
            m_idx = str(idxs[0][0])

            raw_identity = "UNKNOWN"
            if score >= SIM_THRESHOLD and m_idx in id_map:
                raw_identity = id_map[m_idx]

            tracker.report_identity(track_id, raw_identity)

        identity = tracker.current_identity(track_id)
        tracks_out.append({
            "track_id": track_id,
            "bbox": scaled_bbox,
            "identity_id": identity,
            "is_recognized": identity != "UNKNOWN",
            "profile": profiles.get(identity) if identity != "UNKNOWN" else None,
        })

    tracker.cleanup()
    return tracks_out


@app.websocket("/ws/hud")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] React HUD Client Connected.")
    loop = asyncio.get_event_loop()
    tracker = FaceTracker()

    try:
        while True:
            data_str = await websocket.receive_text()

            try:
                data = json.loads(data_str)
                frame_raw = data.get('frame', '')
                if not frame_raw:
                    continue
                if ',' in frame_raw:
                    frame_raw = frame_raw.split(',')[1]

                img_bytes = base64.b64decode(frame_raw)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    continue

                orig_w = data.get('orig_w', 1280)
                orig_h = data.get('orig_h', 720)

                smoothed_tracks = await loop.run_in_executor(
                    None, run_inference, frame, orig_w, orig_h, tracker
                )

                await websocket.send_text(json.dumps({
                    "frame_width": orig_w,
                    "frame_height": orig_h,
                    "tracks": smoothed_tracks,
                }))

            except Exception as frame_err:
                print(f"[Frame Processing Warning] {frame_err}")
                continue

    except WebSocketDisconnect:
        print("[WebSocket] HUD Client Disconnected.")
    except Exception as fatal_err:
        print(f"[WebSocket Fatal Error] {fatal_err}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=30
    )