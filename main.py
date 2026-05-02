"""
============================================================
MAIN.PY — Jalankan Semua Bagian Sekaligus
============================================================
Cara jalankan:
  python main.py

Atau jalankan per bagian:
  python bagian1_thresholding.py
  python bagian2_region_growing.py
  python bagian3_deteksi_tepi.py
  python bagian4_kmeans.py
  python bagian5_watershed.py
  python bagian6_evaluasi.py
============================================================
"""
import numpy as np
from utils import buat_citra_sintetis

from bagian1_thresholding  import demo_thresholding
from bagian2_region_growing import demo_region_growing
from bagian3_deteksi_tepi  import demo_deteksi_tepi
from bagian4_kmeans        import demo_kmeans
from bagian5_watershed     import demo_watershed
from bagian6_evaluasi      import demo_evaluasi


def main():
    print("\n" + "="*55)
    print(" PENGOLAHAN CITRA — SEGMENTASI CITRA")
    print(" Implementasi Python + OpenCV")
    print("="*55)

    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)

    print(f"\n  Citra sintetis dibuat: {img.shape}, dtype={img.dtype}")
    print(f"  Intensitas: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")

    # Jalankan semua demo secara berurutan
    demo_thresholding(img)
    demo_region_growing(img)
    demo_deteksi_tepi(img)
    demo_kmeans(img)
    demo_watershed(img)
    demo_evaluasi(img)

    print("\n" + "="*55)
    print(" SELESAI — Semua demo berhasil dijalankan.")
    print(" File output tersimpan di folder: output_gambar/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
