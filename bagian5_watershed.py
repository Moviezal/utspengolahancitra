"""
============================================================
BAGIAN 5 — WATERSHED SEGMENTATION
============================================================
Topik:
  Segmentasi Watershed berbasis marker.

Langkah:
  1. Thresholding → biner
  2. Morphological operations → tentukan foreground pasti
  3. Distance transform → temukan pusat objek
  4. Peak local max → marker seed
  5. Watershed → batas region

Cara jalankan:
  python bagian5_watershed.py

Output gambar:
  output_gambar/out_5_watershed.png
============================================================
"""
import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy import ndimage
from utils import buat_citra_sintetis, tampilkan_hasil, path_output


def demo_watershed(img):
    """
    Segmentasi Watershed berbasis marker.
    """
    print("\n" + "="*55)
    print(" BAGIAN 5 — WATERSHED SEGMENTATION")
    print("="*55)

    # Step 1: Otsu thresholding
    _, biner = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Morphological opening untuk menghilangkan noise
    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel, iterations=2)

    # Tentukan background pasti
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 3: Distance transform
    dist_tf = ndimage.distance_transform_edt(opening)

    # Step 4: Peak local max → markers
    coords    = peak_local_max(dist_tf, min_distance=15, labels=opening)
    mask_peak = np.zeros(dist_tf.shape, dtype=bool)
    mask_peak[tuple(coords.T)] = True
    markers   = label(mask_peak)
    n_markers = markers.max()
    print(f"\n  Objek terdeteksi (markers): {n_markers}")

    # Step 5: Watershed
    ws_labels = watershed(-dist_tf, markers, mask=opening)

    # Buat visualisasi overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(1, ws_labels.max() + 1):
        warna = tuple(int(c) for c in np.random.randint(80, 230, 3))
        overlay[ws_labels == i] = warna

    # Tandai batas watershed
    batas = np.zeros_like(img)
    for i in range(1, ws_labels.max() + 1):
        kontur, _ = cv2.findContours(
            (ws_labels == i).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, kontur, -1, (255, 255, 255), 1)

    print(f"  Jumlah region hasil watershed: {ws_labels.max()}")

    tampilkan_hasil(
        "Watershed Segmentation — Tahap demi Tahap",
        [img, biner, opening, dist_tf, mask_peak.astype(np.uint8)*255, overlay],
        [
            "1. Citra Asli",
            "2. Otsu Threshold",
            "3. Morphological Opening",
            "4. Distance Transform",
            "5. Peak Markers",
            "6. Hasil Watershed"
        ],
        cmap_list=['gray', 'gray', 'gray', 'hot', 'gray', None],
        simpan=path_output('out_5_watershed.png')
    )


if __name__ == "__main__":
    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)
    demo_watershed(img)
