from pathlib import Path
import cv2
import numpy as np

# ---------- SETTINGS ----------
INPUT_DIR = Path("pages")   # where the big notebook scans are
OUTPUT_DIR = Path("icons")  # where individual PNGs will go
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_AREA = 3000   # ignore tiny dust/noise; increase if tiny specks are saved
PADDING = 15      # extra pixels around each doodle
# -------------------------------

def process_page(img_path, page_index):
    print(f"Processing {img_path.name}...")
    # Read image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read {img_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold for better contrast - makes crisp edges
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 8
    )

    # Invert: now black ink = 0 (black), white paper = 255 (white)
    # We want ink = 255 for processing
    binary_inv = cv2.bitwise_not(binary)

    # Remove tiny noise with morphological opening
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    # Close small gaps in lines
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find connected components (each doodle)
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0
    h_img, w_img = img.shape[:2]

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue  # skip dust

        x, y, w, h = cv2.boundingRect(c)

        # Add padding and clamp to image bounds
        x0 = max(x - PADDING, 0)
        y0 = max(y - PADDING, 0)
        x1 = min(x + w + PADDING, w_img)
        y1 = min(y + h + PADDING, h_img)

        # Crop the mask
        crop_mask = clean[y0:y1, x0:x1]

        # Create a pure black and white image
        # Create white background
        crop_h, crop_w = crop_mask.shape
        crisp_img = np.ones((crop_h, crop_w, 3), dtype=np.uint8) * 255

        # Make ink areas pure black
        crisp_img[crop_mask > 0] = [0, 0, 0]

        # Create alpha channel: where ink exists = opaque, white background = transparent
        alpha = crop_mask.copy()

        # Split into B, G, R channels
        b, g, r = cv2.split(crisp_img)

        # Merge into BGRA with alpha
        rgba = cv2.merge([b, g, r, alpha])

        out_name = OUTPUT_DIR / f"{img_path.stem}_p{page_index:02d}_{count:02d}.png"
        cv2.imwrite(str(out_name), rgba)
        count += 1

    print(f"Saved {count} icons from {img_path.name}.")


def main():
    # Process all jpg/jpeg/png in INPUT_DIR
    image_files = sorted(
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
        + list(INPUT_DIR.glob("*.png"))
    )

    if not image_files:
        print("No images found in 'pages/'")
        return

    for i, img_path in enumerate(image_files):
        process_page(img_path, i)


if __name__ == "__main__":
    main()
