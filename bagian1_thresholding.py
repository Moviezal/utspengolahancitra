# -*- coding: utf-8 -*-
"""
============================================================
BAGIAN 1 - THRESHOLDING
============================================================
Topik:
  a) Global (manual)
  b) Metode Otsu (otomatis)
  c) Thresholding Adaptif (Gaussian)

Cara jalankan:
  python bagian1_thresholding.py

Output gambar:
  output_gambar/out_1_histogram.png
  output_gambar/out_1_thresholding.png
============================================================
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import buat_citra_sintetis, tampilkan_hasil, path_output


def _hitung_otsu_manual(img):
    """
    Implementasi manual algoritma Otsu untuk keperluan pembelajaran.
    Memaksimalkan variansi antar kelas (between-class variance).

    Formula:
      sigma²_B(T) = omega_0 * omega_1 * (mu_0 - mu_1)²
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    total      = img.size
    sigma2_max = 0
    T_optimal  = 0
    sum_total  = np.dot(np.arange(256), hist)
    sum_bg     = 0.0
    w_bg       = 0

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mu_bg   = sum_bg / w_bg
        mu_fg   = (sum_total - sum_bg) / w_fg
        sigma2  = (w_bg / total) * (w_fg / total) * (mu_bg - mu_fg) ** 2
        if sigma2 > sigma2_max:
            sigma2_max = sigma2
            T_optimal  = t

    return T_optimal, sigma2_max


def demo_thresholding(img):
    """
    Mendemonstrasikan tiga jenis thresholding:
      a) Global (manual)
      b) Metode Otsu (otomatis)
      c) Thresholding Adaptif (Gaussian)
    """
    print("\n" + "="*55)
    print(" BAGIAN 1 - THRESHOLDING")
    print("="*55)

    # ── (a) Thresholding Global Manual ──
    T_manual = 128
    _, thresh_global = cv2.threshold(img, T_manual, 255, cv2.THRESH_BINARY)
    print(f"\n[1a] Thresholding Global -> T = {T_manual}")

    # ── (b) Metode Otsu ──
    T_otsu, thresh_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"[1b] Metode Otsu -> T optimal = {T_otsu:.1f} (dihitung otomatis)")

    # Otsu manual untuk memahami mekanisme
    T_hitung, sigma2_max = _hitung_otsu_manual(img)
    print(f"     Verifikasi manual -> T = {T_hitung}, sigma^2_B_maks = {sigma2_max:.2f}")

    # ── (c) Thresholding Adaptif ──
    thresh_adaptif = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # ukuran neighbourhood (harus ganjil)
        C=5             # konstanta pengurang
    )
    print("[1c] Thresholding Adaptif -> blockSize=31, C=5")

    # ── Histogram + garis threshold ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle("Histogram & Threshold", fontweight='bold')
    axes[0].hist(img.ravel(), bins=256, range=(0, 255), color='steelblue', alpha=0.7)
    axes[0].axvline(T_manual, color='orange', linewidth=1.5, label=f'Global T={T_manual}')
    axes[0].axvline(T_otsu,   color='red',    linewidth=1.5, label=f'Otsu T={T_otsu:.0f}')
    axes[0].set_title("Histogram Intensitas")
    axes[0].legend(fontsize=8)
    axes[1].imshow(img, cmap='gray')
    axes[1].set_title("Citra Asli")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(path_output('out_1_histogram.png'), dpi=110, bbox_inches='tight')
    print(f"  [Gambar disimpan: {path_output('out_1_histogram.png')}]")
    plt.show()

    tampilkan_hasil(
        "Thresholding - Perbandingan Metode",
        [img, thresh_global, thresh_otsu, thresh_adaptif],
        [
            "Citra Asli (Grayscale)",
            f"Global T={T_manual}",
            f"Otsu T={T_otsu:.0f} (auto)",
            "Adaptif (Gaussian)"
        ],
        simpan=path_output('out_1_thresholding.png')
    )

    return thresh_otsu


if __name__ == "__main__":
    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)
    demo_thresholding(img)