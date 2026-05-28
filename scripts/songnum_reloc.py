#!/usr/bin/env python3
"""
Song number relocation pass for SFOG hymnal images.

PIPELINE per image:
  1. Extract 1275px source from git d3c989d
  2. vcrop: trim bottom whitespace
  3. Detect song# bbox in source (check BOTH corners, no parity assumption)
     - Outer zone: 350px from each side (avoids centered title)
     - Dilation: 5 iterations to connect digit strokes
     - Union of valid components with max total width 250px
  4. Erase song# from source (paint region white, 5px padding)
  5. Run body-band-aware 800px hcrop (same logic as before)
  6. Paste song# into FIXED position: top-right of cropped output at (800-20-sn_w, 20)
  7. Save

GOAL: All 280 pages have song number at consistent top-right position in 800px output.
Falls back to PIL text rendering if detection fails (0/280 pages needed this).
"""
import os, sys, subprocess
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage

REPO_DIR = os.path.expanduser('~/.openclaw/workspace/scratch/sfog-recrop-2026-05-28/repo')
SFOG_DIR = os.path.join(REPO_DIR, 'resources/images/sfog')
LOG_PATH = os.path.expanduser('~/.openclaw/workspace/scratch/sfog-recrop-2026-05-28/songnum_reloc.log')
SOURCE_COMMIT = 'd3c989d'

# vcrop params
INK_THRESHOLD = 200
MIN_INK_PER_ROW = 3
BOTTOM_PADDING = 40
MIN_RATIO = 0.30

# hcrop params
TOP_BAND_HEIGHT = 80
MIN_INK_COLS = 2
TARGET_WIDTH = 800
TARGET_HALF = TARGET_WIDTH // 2
FALLBACK_PAD = 40

# Song number detection params
SONGNUM_TOP_H = 130       # search in top 130px of source
SONGNUM_OUTER_W = 350     # search in outer 350px (tight enough to avoid centered title)
SONGNUM_MIN_H = 10        # minimum component height
SONGNUM_MAX_H = 100       # maximum component height
SONGNUM_MIN_W = 5         # minimum component width
SONGNUM_MAX_COMP_W = 200  # max per-component width
SONGNUM_MAX_UNION_W = 250 # max total union width (safety against title bleed)
SONGNUM_MAX_Y = 80        # component must start in top 80px
DILATE_ITER = 5           # dilation iterations to connect digit strokes
VERTICAL_GROUP_GAP = 20   # max vertical gap between components in same number group

# Paste position in 800px output: top-right corner
PASTE_PAD = 20            # pixels from top and right edge


def extract_from_git(name, repo_dir, commit):
    rel_path = f'resources/images/sfog/{name}'
    result = subprocess.run(
        ['git', 'show', f'{commit}:{rel_path}'],
        capture_output=True, cwd=repo_dir
    )
    if result.returncode != 0:
        return None
    return Image.open(BytesIO(result.stdout))


def vcrop(img):
    """Trim bottom whitespace from image."""
    arr = np.asarray(img.convert('L'))
    src_h, src_w = arr.shape
    ink_per_row = (arr < INK_THRESHOLD).sum(axis=1)
    rows_with_ink = np.where(ink_per_row >= MIN_INK_PER_ROW)[0]
    if len(rows_with_ink) == 0:
        return img, src_h, 'VCROP-noink'
    last_content = int(rows_with_ink[-1])
    vcrop_h = min(src_h, last_content + 1 + BOTTOM_PADDING)
    ratio = vcrop_h / src_h
    if ratio < MIN_RATIO:
        return img, src_h, 'VCROP-ratio-skip'
    return img.crop((0, 0, src_w, vcrop_h)), vcrop_h, 'OK'


def get_best_group_bbox(zone_arr, x_offset):
    """
    Find the union bbox of the best group of vertically-proximate valid components.
    Groups components by vertical proximity (gap <= VERTICAL_GROUP_GAP) and picks
    the group with the most total ink area. This prevents the bounding box from
    spanning across distant elements (e.g., a title ')' above the actual number).
    """
    ink_mask = (zone_arr < INK_THRESHOLD).astype(np.uint8)
    if ink_mask.sum() == 0:
        return None
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(ink_mask, structure=struct, iterations=DILATE_ITER)
    labeled, n_comps = ndimage.label(dilated)
    valid_comps = []
    for lid in range(1, n_comps + 1):
        comp_mask = (labeled == lid)
        rows = np.where(comp_mask.any(axis=1))[0]
        cols = np.where(comp_mask.any(axis=0))[0]
        if len(rows) == 0:
            continue
        y1, y2 = int(rows[0]), int(rows[-1])
        lx1, lx2 = int(cols[0]), int(cols[-1])
        ch = y2 - y1 + 1
        cw = lx2 - lx1 + 1
        if (SONGNUM_MIN_H <= ch <= SONGNUM_MAX_H and
                SONGNUM_MIN_W <= cw <= SONGNUM_MAX_COMP_W and
                y1 <= SONGNUM_MAX_Y):
            valid_comps.append({'x1': lx1 + x_offset, 'y1': y1,
                                 'x2': lx2 + x_offset, 'y2': y2,
                                 'area': ch * cw})
    if not valid_comps:
        return None
    # Group by vertical proximity (sort by y1, merge if gap <= VERTICAL_GROUP_GAP)
    valid_comps.sort(key=lambda c: c['y1'])
    groups = []
    current_group = [valid_comps[0]]
    for c in valid_comps[1:]:
        group_y2 = max(cc['y2'] for cc in current_group)
        if c['y1'] - group_y2 <= VERTICAL_GROUP_GAP:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)
    # Pick the group with the most total ink area
    best_group = max(groups, key=lambda g: sum(c['area'] for c in g))
    ux1 = min(c['x1'] for c in best_group)
    uy1 = min(c['y1'] for c in best_group)
    ux2 = max(c['x2'] for c in best_group)
    uy2 = max(c['y2'] for c in best_group)
    # Safety check: reject if union is too wide (would indicate title bleed-in)
    if ux2 - ux1 > SONGNUM_MAX_UNION_W:
        return None
    return (ux1, uy1, ux2 + 1, uy2 + 1)


def detect_song_number(img):
    """
    Detect song number bbox in source image.
    Checks BOTH corners (no parity assumption - book layout varies).
    Returns (x1, y1, x2, y2, side) or (None, None, None, None, 'none').
    """
    w, h = img.size
    arr = np.asarray(img.convert('L'))
    top_band = arr[:SONGNUM_TOP_H, :]
    lc = get_best_group_bbox(top_band[:, :SONGNUM_OUTER_W], 0)
    rc = get_best_group_bbox(top_band[:, w - SONGNUM_OUTER_W:], w - SONGNUM_OUTER_W)
    if lc and not rc:
        return (*lc, 'left')
    elif rc and not lc:
        return (*rc, 'right')
    elif lc and rc:
        # Both sides detected - pick the one closer to the outer edge
        l_dist = lc[0]
        r_dist = w - rc[2]
        return (*lc, 'left') if l_dist <= r_dist else (*rc, 'right')
    return (None, None, None, None, 'none')


def hcrop(img_vcropped):
    """Body-band-aware 800px horizontal crop."""
    w, h = img_vcropped.size
    arr = np.asarray(img_vcropped.convert('L'))
    body = arr[TOP_BAND_HEIGHT:, :]
    body_ink_cols = np.where((body < INK_THRESHOLD).sum(axis=0) >= MIN_INK_COLS)[0]
    if len(body_ink_cols) < 5:
        body_left = FALLBACK_PAD
        body_right = w - FALLBACK_PAD
        status = 'HCROP-fallback'
    else:
        body_left = int(body_ink_cols[0])
        body_right = int(body_ink_cols[-1])
        status = 'OK'
    body_center = (body_left + body_right) // 2
    crop_left = max(0, body_center - TARGET_HALF)
    crop_right = min(w, body_center + TARGET_HALF)
    if crop_right - crop_left < TARGET_WIDTH:
        if crop_left == 0:
            crop_right = min(w, TARGET_WIDTH)
        else:
            crop_left = max(0, w - TARGET_WIDTH)
    cropped = img_vcropped.crop((crop_left, 0, crop_right, h))
    return cropped, body_left, body_right, crop_left, crop_right, status


def make_fallback_number(song_num, ref_height=35):
    """Render song number as PIL text (fallback when detection fails)."""
    font = None
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, ref_height)
            break
    if font is None:
        font = ImageFont.load_default()
    text = str(song_num)
    tmp = Image.new('RGB', (300, 100), (255, 255, 255))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    num_img = Image.new('RGB', (tw + 4, th + 4), (255, 255, 255))
    draw = ImageDraw.Draw(num_img)
    draw.text((2 - bbox[0], 2 - bbox[1]), text, fill=(0, 0, 0), font=font)
    return num_img


songs = sorted([f'SFOG_{i:03d}.png' for i in range(1, 281)])
print(f"Processing {len(songs)} images from commit {SOURCE_COMMIT}")

results = []
fallbacks = []

with open(LOG_PATH, 'w') as logf:
    logf.write("song,src_w,src_h,vcrop_h,sn_x1,sn_y1,sn_x2,sn_y2,sn_side,sn_method,sn_w,sn_h,body_left,body_right,crop_left,crop_right,out_w,out_h,status\n")

    for name in songs:
        song_num = int(name.replace('SFOG_', '').replace('.png', ''))

        img = extract_from_git(name, REPO_DIR, SOURCE_COMMIT)
        if img is None:
            print(f"  SKIP {name}: git show failed")
            logf.write(f"{name},,,,,,,,,,,,,,,,,SKIP-git\n")
            continue

        src_w, src_h = img.size

        # vcrop
        img_v, vcrop_h, vcrop_status = vcrop(img)
        if vcrop_status != 'OK':
            img_v = img.copy()
            vcrop_h = src_h

        # Detect song number in original source
        sn_x1, sn_y1, sn_x2, sn_y2, sn_side = detect_song_number(img)

        if sn_x1 is not None:
            sn_patch = img.crop((sn_x1, sn_y1, sn_x2, sn_y2)).copy()
            sn_method = 'detected'
        else:
            sn_patch = make_fallback_number(song_num)
            sn_method = 'fallback-text'
            sn_x1 = sn_y1 = sn_x2 = sn_y2 = -1
            fallbacks.append(name)
            print(f"  FALLBACK {name}: no detection, using text render")

        # Erase song number from source before hcrop, UNLESS the number is in the body
        # detection zone (sn_y1 >= TOP_BAND_HEIGHT-10). If the number is in the body band,
        # erasing it from source will shift the body center and may misplace the crop window.
        # In that case, we don't erase from source - instead we'll erase from the final frame
        # after hcrop.
        erase_in_frame = (sn_x1 >= 0 and sn_y1 >= TOP_BAND_HEIGHT - 10)
        if sn_x1 >= 0 and not erase_in_frame:
            img_v_arr = np.array(img_v)
            ey1 = max(0, sn_y1 - 5)
            ey2 = min(vcrop_h, sn_y2 + 5)
            ex1 = max(0, sn_x1 - 5)
            ex2 = min(src_w, sn_x2 + 5)
            img_v_arr[ey1:ey2, ex1:ex2] = 255
            img_v = Image.fromarray(img_v_arr)

        # hcrop
        final_img, body_left, body_right, crop_left, crop_right, hcrop_status = hcrop(img_v)

        # For numbers in the body zone: erase from final frame (not source), then paste
        if erase_in_frame:
            final_arr = np.array(final_img)
            fx1 = max(0, sn_x1 - crop_left - 3)
            fx2 = min(TARGET_WIDTH, sn_x2 - crop_left + 3)
            fy1 = max(0, sn_y1 - 3)
            fy2 = min(vcrop_h, sn_y2 + 3)
            final_arr[fy1:fy2, fx1:fx2] = 255
            final_img = Image.fromarray(final_arr)

        # Paste song number into top-right corner
        sn_w, sn_h = sn_patch.size
        paste_x = TARGET_WIDTH - PASTE_PAD - sn_w
        paste_y = PASTE_PAD
        final_img.paste(sn_patch, (paste_x, paste_y))

        out_w, out_h = final_img.size

        # Save
        out_path = os.path.join(SFOG_DIR, name)
        final_img.save(out_path, optimize=False)

        combined_status = hcrop_status
        if vcrop_status != 'OK':
            combined_status = vcrop_status
        if sn_method == 'fallback-text':
            combined_status += '-fallback'

        logf.write(f"{name},{src_w},{src_h},{vcrop_h},{sn_x1},{sn_y1},{sn_x2},{sn_y2},{sn_side},{sn_method},{sn_w},{sn_h},{body_left},{body_right},{crop_left},{crop_right},{out_w},{out_h},{combined_status}\n")
        results.append(dict(
            name=name, song_num=song_num, src_w=src_w, src_h=src_h,
            vcrop_h=vcrop_h, sn_side=sn_side, sn_method=sn_method,
            sn_w=sn_w, sn_h=sn_h, out_w=out_w, out_h=out_h,
            status=combined_status
        ))

        if len(results) % 50 == 0:
            print(f"  {len(results)}/280 done...")

ok = [r for r in results if r['status'] == 'OK']
detected = [r for r in results if r['sn_method'] == 'detected']
fallback = [r for r in results if r['sn_method'] == 'fallback-text']
widths = set(r['out_w'] for r in results)

print(f"\nDone: {len(results)} processed")
print(f"  OK: {len(ok)}")
print(f"  Detected from source: {len(detected)}")
print(f"  Fallback text render: {len(fallback)}")
if fallback:
    for r in [r for r in results if r['sn_method'] == 'fallback-text']:
        print(f"    {r['name']}: {r['status']}")
print(f"Output widths: {sorted(widths)}")
print(f"Log: {LOG_PATH}")
