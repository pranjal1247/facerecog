import os
import sys
import json
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# ==========================================
# 1. PATH CONFIGURATION & CHECKS
# ==========================================
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.json")
PROFILES_PATH = os.path.join(DATA_DIR, "profiles.json")

SIMILARITY_THRESHOLD = 0.50

# Check required FAISS artifacts exist
for path in [FAISS_INDEX_PATH, ID_MAP_PATH, PROFILES_PATH]:
    if not os.path.exists(path):
        print(f"[Error] Required artifact missing: '{path}'")
        print("Please run `python generate_embeddings.py` first!")
        sys.exit(1)

# ==========================================
# 2. LOAD FAISS INDEX & METADATA
# ==========================================
print("[Init] Loading FAISS Index and Metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(ID_MAP_PATH, "r") as f:
    id_map = json.load(f)

with open(PROFILES_PATH, "r") as f:
    profiles = json.load(f)

print(f"[Init] Loaded FAISS index with {index.ntotal} total vectors.")

# ==========================================
# 3. INITIALIZE INSIGHTFACE
# ==========================================
print("[Init] Loading InsightFace Verification Engine...")
app = FaceAnalysis(
    name='buffalo_l', 
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# 4. RESOLVE TEST IMAGE PATH
# ==========================================
if len(sys.argv) > 1:
    test_img_path = sys.argv[1]
else:
    test_img_path = input("\nEnter path to test image (e.g., test.jpg): ").strip()

if not os.path.exists(test_img_path):
    print(f"[Error] Image not found at path: '{test_img_path}'")
    sys.exit(1)

# ==========================================
# 5. RUN VERIFICATION & VECTOR SEARCH
# ==========================================
img = cv2.imread(test_img_path)
if img is None:
    print(f"[Error] Could not decode image file: '{test_img_path}'")
    sys.exit(1)

print(f"\n[Processing] Analyzing '{test_img_path}'...")
faces = app.get(img)

if len(faces) == 0:
    print("[Result] No faces detected in the provided image.")
    sys.exit(0)

print(f"[Processing] Detected {len(faces)} face(s). Searching index...\n")

for i, face in enumerate(faces):
    bbox = [int(coord) for coord in face.bbox]
    x1, y1, x2, y2 = bbox

    # L2 Normalize embedding for Cosine Similarity search
    emb = face.embedding
    emb_norm = (emb / np.linalg.norm(emb)).astype('float32').reshape(1, -1)

    # Search top 1 nearest neighbor
    sims, idxs = index.search(emb_norm, k=1)
    score = float(sims[0][0])
    match_idx = str(idxs[0][0])

    identity = "UNKNOWN"
    profile = None

    if score >= SIMILARITY_THRESHOLD and match_idx in id_map:
        identity = id_map[match_idx]
        profile = profiles.get(identity, {})
        is_matched = True
    else:
        is_matched = False

    # Output verification details
    print(f"--- Face #{i+1} ---")
    print(f"  Bounding Box : [{x1}, {y1}, {x2}, {y2}]")
    print(f"  Similarity   : {score:.4f} (Threshold: {SIMILARITY_THRESHOLD})")
    
    if is_matched:
        print(f"  Status       : MATCH CONFIRMED")
        print(f"  Identity ID  : {identity}")
        print(f"  Name         : {profile.get('name', 'N/A')}")
        print(f"  Title        : {profile.get('title', 'N/A')}")
        print(f"  Department   : {profile.get('department', 'N/A')}\n")
    else:
        print(f"  Status       : UNKNOWN / UNRECOGNIZED\n")

    # Annotate image for visual inspection
    box_color = (0, 255, 0) if is_matched else (0, 165, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    
    label = f"{profile.get('name', 'UNKNOWN')} ({score:.2f})" if is_matched else f"UNKNOWN ({score:.2f})"
    cv2.putText(img, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

# Display annotated result window
WINDOW_NAME = "FAISS Verification Test"
cv2.imshow(WINDOW_NAME, img)
print("Press any key on the image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()