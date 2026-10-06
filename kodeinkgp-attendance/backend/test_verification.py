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
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.json")
PROFILES_PATH = os.path.join(DATA_DIR, "profiles.json")
LOGO_PATH = os.path.join(DATA_DIR, "kodeinkgp_logo.png")

SIMILARITY_THRESHOLD = 0.50

from PIL import Image, ImageDraw, ImageFont

# ==========================================
# EXACT HUD THEME COLORS (converted from Tailwind hex to RGB)
# ==========================================
CYBER_CYAN = (0, 240, 255)      # #00f0ff
CYBER_GREEN = (0, 255, 153)     # #00ff99
CYBER_AMBER = (255, 153, 0)     # #ff9900
WHITE = (255, 255, 255)
CARD_BG = (0, 0, 0)             # black, blended with alpha below

def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])

COLOR_RECOGNIZED = bgr(CYBER_GREEN)
COLOR_UNKNOWN = bgr(CYBER_AMBER)
COLOR_CYAN = bgr(CYBER_CYAN)

# ==========================================
# FONT SETUP — must be a real monospace font to match ui-monospace/Menlo/Consolas
# ==========================================
FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",       # Consolas (Windows) — best match
    "C:/Windows/Fonts/cascadiacode.ttf",  # Cascadia Code (Windows Terminal font)
    "/System/Library/Fonts/Menlo.ttc",    # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux fallback
]

def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    print("[Warning] No monospace font file found — falling back to default (quality will differ).")
    return ImageFont.load_default()

FONT_LABEL = load_font(20)
FONT_NAME = load_font(28)
FONT_BODY = load_font(18)
FONT_QUOTE = load_font(17)


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_label_tag_pil(draw, text, x, y, color_rgb):
    """Filled tag above the box, black text, matching the React label pill."""
    bbox = draw.textbbox((0, 0), text, font=FONT_LABEL)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 8, 5

    tag_top = y - th - pad_y * 2
    draw.rectangle(
        [x, tag_top, x + tw + pad_x * 2, y],
        fill=color_rgb
    )
    draw.text((x + pad_x, tag_top + pad_y - 2), text, font=FONT_LABEL, fill=(0, 0, 0))


def wrap_text_pil(draw, text, font, max_width):
    words = text.split(' ')
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def draw_info_card_pil(base_img, x, y, profile, border_rgb):
    """Composites a semi-transparent card with monospace text, matching
    the React popup's bg-black/80 backdrop-blur + font-mono styling."""
    card_w = 380
    pad = 20
    line_gap = 12

    pil_img = cv2_to_pil(base_img)
    draw = ImageDraw.Draw(pil_img)

    entries = []
    entries.append((profile.get("name", "N/A"), FONT_NAME, WHITE, 16))
    if profile.get("department"):
        entries.append((profile["department"], FONT_BODY, CYBER_CYAN, 10))
    if profile.get("role"):
        entries.append((profile["role"], FONT_BODY, CYBER_CYAN, 16))
    if profile.get("fun_fact"):
        wrapped = wrap_text_pil(draw, f'"{profile["fun_fact"]}"', FONT_QUOTE, card_w - pad * 2)
        for i, wline in enumerate(wrapped):
            entries.append((wline, FONT_QUOTE, CYBER_AMBER, 6 if i < len(wrapped) - 1 else 0))

    card_h = pad
    for text, font, color, gap in entries:
        bbox = draw.textbbox((0, 0), text, font=font)
        card_h += (bbox[3] - bbox[1]) + gap
    card_h += pad

    img_w = pil_img.width
    flip_left = (x + card_w) > img_w
    card_x = x - card_w - 16 if flip_left else x

    # Semi-transparent black background via a separate RGBA overlay
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [card_x, y, card_x + card_w, y + card_h],
        fill=(0, 0, 0, 210)
    )
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    draw.rectangle([card_x, y, card_x + card_w, y + card_h], outline=border_rgb, width=1)

    cursor_y = y + pad
    for text, font, color, gap in entries:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((card_x + pad, cursor_y), text, font=font, fill=color)
        cursor_y += (bbox[3] - bbox[1]) + gap

    return pil_to_cv2(pil_img)

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
# 4. DRAWING HELPERS (HUD-style rendering)
# ==========================================
def overlay_transparent(bg, overlay_rgba, x, y, target_width=None):
    """Alpha-blend a BGRA logo onto the frame at (x, y), matching how the
    logo renders with no background in the React version."""
    if overlay_rgba.shape[2] != 4:
        # No alpha channel available; just paste as-is
        oh, ow = overlay_rgba.shape[:2]
        bg[y:y+oh, x:x+ow] = overlay_rgba
        return bg

    if target_width is not None:
        scale = target_width / overlay_rgba.shape[1]
        overlay_rgba = cv2.resize(
            overlay_rgba, (target_width, int(overlay_rgba.shape[0] * scale))
        )

    oh, ow = overlay_rgba.shape[:2]
    bh, bw = bg.shape[:2]

    if y + oh > bh or x + ow > bw:
        oh = min(oh, bh - y)
        ow = min(ow, bw - x)
        overlay_rgba = overlay_rgba[:oh, :ow]

    overlay_bgr = overlay_rgba[:, :, :3]
    alpha = (overlay_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]

    roi = bg[y:y+oh, x:x+ow]
    bg[y:y+oh, x:x+ow] = (alpha * overlay_bgr + (1 - alpha) * roi).astype(np.uint8)
    return bg


def draw_label_tag(img, text, x, y, color):
    """Small filled tag above the box, e.g. 'Pranjal', black text on colored bg."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6

    cv2.rectangle(img, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def wrap_text(text, font, scale, thickness, max_width):
    """Wraps text to fit within max_width pixels, for the fun-fact quote."""
    words = text.split(' ')
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def draw_info_card(img, x, y, profile, box_color):
    """Floating info card to the right of the box (or left if it would
    overflow), mirroring the React popup: Name, Department, Role, Fun Fact."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    card_w = 340
    pad = 16
    line_gap = 10

    lines = []  # (text, scale, thickness, color, extra_gap_after)
    lines.append((profile.get("name", "N/A"), 0.8, 2, (255, 255, 255), 14))
    if profile.get("department"):
        lines.append((profile["department"], 0.55, 1, COLOR_CYAN, 8))
    if profile.get("role"):
        lines.append((profile["role"], 0.55, 1, COLOR_CYAN, 14))
    if profile.get("fun_fact"):
        wrapped = wrap_text(f'"{profile["fun_fact"]}"', font, 0.5, 1, card_w - pad * 2)
        for i, wline in enumerate(wrapped):
            lines.append((wline, 0.5, 1, (0, 200, 255), 6 if i < len(wrapped) - 1 else 0))

    # Compute card height from accumulated line heights
    card_h = pad
    for text, scale, thickness, color, gap in lines:
        (_, th), _ = cv2.getTextSize(text, font, scale, thickness)
        card_h += th + gap
    card_h += pad

    # Flip to the left of the box if it would overflow the right edge
    img_w = img.shape[1]
    flip_left = (x + card_w) > img_w
    card_x = x - card_w - 16 if flip_left else x

    # Semi-transparent black background (matches bg-black/80 backdrop-blur)
    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, y), (card_x + card_w, y + card_h), COLOR_CARD_BG, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (card_x, y), (card_x + card_w, y + card_h), box_color, 1)

    cursor_y = y + pad
    for text, scale, thickness, color, gap in lines:
        (_, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cursor_y += th
        cv2.putText(img, text, (card_x + pad, cursor_y), font, scale, color, thickness, cv2.LINE_AA)
        cursor_y += gap

    return img


# ==========================================
# 5. RESOLVE TEST IMAGE PATH
# ==========================================
if len(sys.argv) > 1:
    test_img_path = sys.argv[1]
else:
    test_img_path = input("\nEnter path to test image (e.g., test.jpg): ").strip()

if not os.path.exists(test_img_path):
    print(f"[Error] Image not found at path: '{test_img_path}'")
    sys.exit(1)

# ==========================================
# 6. RUN VERIFICATION & VECTOR SEARCH
# ==========================================
img = cv2.imread(test_img_path)
if img is None:
    print(f"[Error] Could not decode image file: '{test_img_path}'")
    sys.exit(1)

# Downscale large photos to a sane working resolution — this matches
# what the live HUD does with webcam frames, and prevents the OpenCV
# window (and the info card drawn on it) from overflowing off-screen
# on high-resolution phone/webcam photos.
MAX_DISPLAY_WIDTH = 1280
if img.shape[1] > MAX_DISPLAY_WIDTH:
    scale = MAX_DISPLAY_WIDTH / img.shape[1]
    img = cv2.resize(img, (MAX_DISPLAY_WIDTH, int(img.shape[0] * scale)))
    print(f"[Info] Resized image to {img.shape[1]}x{img.shape[0]} for display/processing.")

print(f"\n[Processing] Analyzing '{test_img_path}'...")
faces = app.get(img)

if len(faces) == 0:
    print("[Result] No faces detected in the provided image.")
    sys.exit(0)

print(f"[Processing] Detected {len(faces)} face(s). Searching index...\n")

for i, face in enumerate(faces):
    bbox = [int(coord) for coord in face.bbox]
    x1, y1, x2, y2 = bbox

    emb = face.embedding
    emb_norm = (emb / np.linalg.norm(emb)).astype('float32').reshape(1, -1)

    sims, idxs = index.search(emb_norm, k=1)
    score = float(sims[0][0])
    match_idx = str(idxs[0][0])

    identity = "UNKNOWN"
    profile = None
    is_matched = False

    if score >= SIMILARITY_THRESHOLD and match_idx in id_map:
        identity = id_map[match_idx]
        profile = profiles.get(identity, {})
        is_matched = True

    print(f"--- Face #{i+1} ---")
    print(f"  Bounding Box : [{x1}, {y1}, {x2}, {y2}]")
    print(f"  Similarity   : {score:.4f} (Threshold: {SIMILARITY_THRESHOLD})")
    if is_matched:
        print(f"  Status       : MATCH CONFIRMED")
        print(f"  Identity ID  : {identity}")
        print(f"  Name         : {profile.get('name', 'N/A')}\n")
    else:
        print(f"  Status       : UNKNOWN / UNRECOGNIZED\n")

    box_color_bgr = COLOR_RECOGNIZED if is_matched else COLOR_UNKNOWN
    box_color_rgb = CYBER_GREEN if is_matched else CYBER_AMBER

    # Bounding box (thin OpenCV rectangle is fine, only text quality needed PIL)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color_bgr, 2)

    # Label tag + info card via PIL for crisp monospace text
    pil_img = cv2_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    label_text = profile.get("name", "UNKNOWN") if is_matched else "UNKNOWN"
    draw_label_tag_pil(draw, label_text, x1, y1, box_color_rgb)
    img = pil_to_cv2(pil_img)

    if is_matched and profile:
        img = draw_info_card_pil(img, x2 + 16, y1, profile, box_color_rgb)

# Logo, bottom-left, no background — mirrors the React <img> overlay
if os.path.exists(LOGO_PATH):
    logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if logo is not None:
        logo_target_w = 180
        margin = 20
        logo_h_scaled = int(logo.shape[0] * (logo_target_w / logo.shape[1]))
        img = overlay_transparent(
            img, logo,
            x=margin,
            y=img.shape[0] - logo_h_scaled - margin,
            target_width=logo_target_w
        )
else:
    print(f"[Warning] Logo not found at '{LOGO_PATH}' — skipping overlay.")

# ==========================================
# 7. DISPLAY & SAVE RESULT
# ==========================================
WINDOW_NAME = "FAISS Verification Test - HUD Preview"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, img.shape[1], img.shape[0])
cv2.imshow(WINDOW_NAME, img)

output_path = os.path.splitext(test_img_path)[0] + "_hud_result.png"
cv2.imwrite(output_path, img)
print(f"[Saved] Annotated result written to '{output_path}'")

print("Press any key on the image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()