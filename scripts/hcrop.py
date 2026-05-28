#!/usr/bin/env python3
"""
hcrop.py - Body-band-aware horizontal recrop for SFOG hymnal PNGs.

PROBLEM:
The original horizontal autocrop used the ink bounding box of all content,
including the song number printed in the outer corner of each page. This caused
asymmetric horizontal margins: the body (music notation, chords, lyrics) was
visually off-center by 30-250px, with more white space on the side opposite the
song number.

SOLUTION:
Detect the body content band (skip the top 80px header/song-number zone),
center a fixed 800px window on the body content center. The song number floats
in the page corner and may clip by 1-9px at most (acceptable for 3/280 pages).

ALGORITHM:
  1. Skip top TOP_BAND_HEIGHT rows (header zone: song number + title)
  2. Find leftmost and rightmost ink columns in the body band
  3. Compute body_center = (body_left + body_right) // 2
  4. Crop to [body_center - 400, body_center + 400]  (= 800px window)
  5. Clamp to image bounds if necessary

OUTPUT:
  All images: exactly TARGET_WIDTH (800px) wide, body content centered.
  Heights are preserved from the input (apply vcrop first if needed).

USAGE:
  Set SFOG_DIR to the directory containing SFOG_*.png files.
  Run: python3 hcrop.py
"""
import os, sys, glob
from PIL import Image
import numpy as np

SFOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'resources', 'images', 'sfog')
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hcrop.log')

INK_THRESHOLD = 200       # pixel value < this = "ink"
MIN_INK_COLS = 2          # min rows containing ink for a column to count as body ink
TOP_BAND_HEIGHT = 80      # rows to skip (song number + title header zone)
TARGET_WIDTH = 800        # output width in pixels
TARGET_HALF = TARGET_WIDTH // 2
FALLBACK_PAD = 40         # fallback padding if no body content detected

paths = sorted(glob.glob(os.path.join(SFOG_DIR, 'SFOG_*.png')))
if not paths:
    print(f"ERROR: No SFOG_*.png found in {SFOG_DIR}")
    sys.exit(1)
print(f"Found {len(paths)} PNGs in {SFOG_DIR}")

skipped = []
results = []

with open(LOG, 'w') as logf:
    logf.write("song,orig_w,orig_h,body_left,body_right,body_center,crop_left,crop_right,out_w,status\n")

    for p in paths:
        name = os.path.basename(p)
        img = Image.open(p)
        w, h = img.size
        gray = img.convert('L')
        arr = np.asarray(gray)

        # Detect body content (skip song-number/title header rows)
        body = arr[TOP_BAND_HEIGHT:, :]
        body_ink_cols = np.where((body < INK_THRESHOLD).sum(axis=0) >= MIN_INK_COLS)[0]

        if len(body_ink_cols) < 5:
            # Fallback: no body detected, use image center with padding
            body_left = FALLBACK_PAD
            body_right = w - FALLBACK_PAD
            status = 'FALLBACK-noink'
        else:
            body_left = int(body_ink_cols[0])
            body_right = int(body_ink_cols[-1])
            status = 'OK'

        # Symmetric crop centered on body content
        body_center = (body_left + body_right) // 2
        crop_left = max(0, body_center - TARGET_HALF)
        crop_right = min(w, body_center + TARGET_HALF)

        # If we hit image bounds (image < 800px wide), shift window to fill TARGET_WIDTH
        if crop_right - crop_left < TARGET_WIDTH:
            if crop_left == 0:
                crop_right = min(w, TARGET_WIDTH)
            else:
                crop_left = max(0, w - TARGET_WIDTH)

        out_w = crop_right - crop_left
        cropped = img.crop((crop_left, 0, crop_right, h))
        cropped.save(p)

        logf.write(f"{name},{w},{h},{body_left},{body_right},{body_center},{crop_left},{crop_right},{out_w},{status}\n")
        results.append((name, w, h, body_left, body_right, body_center, crop_left, crop_right, out_w, status))

ok = [r for r in results if r[9] == 'OK']
fallback = [r for r in results if r[9] != 'OK']
widths = set(r[8] for r in results)

print(f"Processed: {len(results)}")
print(f"  OK: {len(ok)}")
print(f"  Fallback (no body ink): {len(fallback)}")
if fallback:
    for r in fallback:
        print(f"    {r[0]}: {r[9]}")
print(f"Output widths: {sorted(widths)}")
print(f"Log: {LOG}")
