"""
============================================================
BAGIAN 4 — SEGMENTASI K-MEANS CLUSTERING
============================================================
Topik:
  Segmentasi menggunakan K-Means clustering.
  Setiap piksel dianggap sebagai titik data 1D (intensitas).
  Algoritma mengelompokkan piksel ke dalam K cluster.

Cara jalankan:
  python bagian4_kmeans.py

Output gambar:
  output_gambar/out_4_kmeans.png
  output_gambar/out_4b_kmeans_color.png
============================================================
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import buat_citra_sintetis, tampilkan_hasil, path_output


def demo_kmeans(img):
    """
    Segmentasi menggunakan K-Means clustering.
    Setiap piksel dianggap sebagai titik data 1D (intensitas).
    Algoritma mengelompokkan piksel ke dalam K cluster.
    """
    print("\n" + "="*55)
    print(" BAGIAN 4 — K-MEANS CLUSTERING")
    print("="*55)

    def segmentasi_kmeans(citra, K):
        """
        Menerapkan K-Means pada citra grayscale.
        Mengembalikan citra tersegmentasi dan label tiap piksel.
        """
        # Reshape menjadi vektor kolom
        data = citra.reshape(-1, 1).astype(np.float32)

        kriteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,    # iterasi maksimum
            0.2     # epsilon
        )
        _, labels, centroid = cv2.kmeans(
            data, K, None, kriteria,
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS
        )
        centroid = np.uint8(centroid)
        hasil    = centroid[labels.flatten()].reshape(citra.shape)
        return hasil, labels.reshape(citra.shape), centroid.flatten()

    nilai_K      = [2, 3, 4, 5]
    gambar_hasil = [img]
    label_hasil  = ["Citra Asli"]

    for K in nilai_K:
        seg, label_map, centroid = segmentasi_kmeans(img, K)
        centroid_str = ', '.join(str(c) for c in sorted(centroid))
        print(f"\n  K={K}: centroid intensitas = [{centroid_str}]")
        gambar_hasil.append(seg)
        label_hasil.append(f"K-Means K={K}")

    tampilkan_hasil(
        "Segmentasi K-Means — Pengaruh Jumlah Cluster (K)",
        gambar_hasil,
        label_hasil,
        simpan=path_output('out_4_kmeans.png')
    )

    # ── Visualisasi label berwarna untuk K=3 ──
    seg3, labels3, centroid3 = segmentasi_kmeans(img, 3)
    color_map = np.array([[220, 80, 80], [80, 200, 120], [80, 130, 220]], dtype=np.uint8)
    colored   = color_map[labels3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("K-Means K=3 — Analisis Detail", fontweight='bold')
    axes[0].imshow(img,    cmap='gray'); axes[0].set_title("Citra Asli");          axes[0].axis('off')
    axes[1].imshow(seg3,   cmap='gray'); axes[1].set_title("Hasil (grayscale)");   axes[1].axis('off')
    axes[2].imshow(colored);            axes[2].set_title("Label per cluster (berwarna)"); axes[2].axis('off')

    legenda = [
        mpatches.Patch(color=[v/255 for v in color_map[i]],
                       label=f"Cluster {i+1} (I≈{centroid3[i]})")
        for i in range(3)
    ]
    axes[2].legend(handles=legenda, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path_output('out_4b_kmeans_color.png'), dpi=110, bbox_inches='tight')
    print(f"  [Gambar disimpan: {path_output('out_4b_kmeans_color.png')}]")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)
    demo_kmeans(img)
