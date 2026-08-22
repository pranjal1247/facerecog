import os
import sys
import json
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# ==========================================
# 1. DYNAMIC PATH RESOLUTION
# ==========================================
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.json")
PROFILES_PATH = os.path.join(DATA_DIR, "profiles.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)

print(f"[Init] Target Dataset Path: '{RAW_IMAGES_DIR}'")

# ==========================================
# 2. INITIALIZE INSIGHTFACE
# ==========================================
print("[Init] Loading InsightFace Embedding Engine...")
app = FaceAnalysis(
    name='buffalo_l', 
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# 3. SCAN & EXTRACT EMBEDDINGS
# ==========================================
embeddings_list = []
id_map = {}

# Load existing profiles so re-running this script never wipes out
# custom edits you've made (name, hall, fun_fact, etc.)
if os.path.exists(PROFILES_PATH):
    with open(PROFILES_PATH, "r") as f:
        profiles = json.load(f)
    print(f"[Init] Loaded {len(profiles)} existing profile(s) — will preserve them.")
else:
    profiles = {}

member_folders = [
    f for f in os.listdir(RAW_IMAGES_DIR) 
    if os.path.isdir(os.path.join(RAW_IMAGES_DIR, f))
]

if not member_folders:
    print(f"\n[Warning] No subfolders found in {RAW_IMAGES_DIR}.")
    sys.exit(1)

idx_counter = 0

for member_id in member_folders:
    member_dir = os.path.join(RAW_IMAGES_DIR, member_id)
    image_files = [f for f in os.listdir(member_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n[Processing] Folder '{member_id}' ({len(image_files)} images found)...")

    # Default profile metadata
        # Only create a default profile if this member doesn't already have one.
    # Existing custom entries (name, hall, fun_fact, etc.) are left untouched.
    if member_id not in profiles:
        profiles[member_id] = {
            "name": member_id.replace("_", " ").title(),
            "title": "IIT KGP Member",
            "department": "Computer Science & Engineering",
            "hall": "",
            "year": "",
            "fun_fact": ""
        }
        print(f"  └─ New member detected, added default profile for '{member_id}'")

    for img_name in image_files:
        img_path = os.path.join(member_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"  └─ Skipping {img_name} (No face detected)")
            continue

        # Extract primary face embedding
        emb = faces[0].embedding
        emb_norm = (emb / np.linalg.norm(emb)).astype('float32')

        embeddings_list.append(emb_norm)
        id_map[str(idx_counter)] = member_id
        idx_counter += 1
        print(f"  └─ Extracted embedding from {img_name}")

if not embeddings_list:
    print("\n[Error] No valid facial embeddings were extracted. Check your image files.")
    sys.exit(1)

# ==========================================
# 4. BUILD & SAVE FAISS INDEX + METADATA
# ==========================================
embeddings_matrix = np.array(embeddings_list).astype('float32')
dimension = embeddings_matrix.shape[1]

# IndexFlatIP calculates Cosine Similarity on normalized vectors
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_matrix)

faiss.write_index(index, FAISS_INDEX_PATH)

with open(ID_MAP_PATH, "w") as f:
    json.dump(id_map, f, indent=4)

with open(PROFILES_PATH, "w") as f:
    json.dump(profiles, f, indent=4)

print(f"\n[Success] FAISS index generated with {index.ntotal} vector(s)!")
print(f"  ├─ Index  : {FAISS_INDEX_PATH}")
print(f"  ├─ ID Map : {ID_MAP_PATH}")
print(f"  └─ Profiles: {PROFILES_PATH}")