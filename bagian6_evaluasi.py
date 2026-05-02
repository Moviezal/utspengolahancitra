"""
============================================================
BAGIAN 6 - EVALUASI SEGMENTASI
============================================================
Topik:
  Menghitung metrik evaluasi standar segmentasi:
  - Pixel Accuracy
  - IoU (Intersection over Union / Jaccard Index)
  - Dice Coefficient
  - Precision & Recall

Cara jalankan:
  python bagian6_evaluasi.py

Output gambar:
  output_gambar/out_6_evaluasi.png
============================================================
"""
import cv2
import numpy as np
from utils import buat_citra_sintetis, tampilkan_hasil, path_output


def demo_evaluasi(img):
    """
    Menghitung metrik evaluasi standar segmentasi:
      - Pixel Accuracy
      - IoU (Intersection over Union / Jaccard Index)
      - Dice Coefficient
      - Precision & Recall
    """
    print("\n" + "="*55)
    print(" BAGIAN 6 - EVALUASI SEGMENTASI")
    print("="*55)

    def pixel_accuracy(pred, gt):
        return np.sum(pred == gt) / gt.size

    def iou(pred, gt):
        interseksi = np.logical_and(pred, gt).sum()
        gabungan   = np.logical_or(pred, gt).sum()
        return interseksi / gabungan if gabungan > 0 else 0.0

    def dice(pred, gt):
        interseksi = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()
        return 2 * interseksi / total if total > 0 else 0.0

    def precision_recall(pred, gt):
        TP   = np.logical_and(pred,  gt).sum()
        FP   = np.logical_and(pred, ~gt).sum()
        FN   = np.logical_and(~pred,  gt).sum()
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return prec, rec

    # Ground truth sintetis (lingkaran)
    gt = np.zeros_like(img, dtype=bool)
    cv2.circle(gt.view(np.uint8), (128, 128), 50, 255, -1)
    gt = gt.astype(bool)

    prediksi_list = []
    label_pred    = []

    # Prediksi A (hampir sempurna)
    p1 = np.zeros_like(img, dtype=bool)
    cv2.circle(p1.view(np.uint8), (130, 130), 48, 255, -1)
    prediksi_list.append(p1.astype(bool))
    label_pred.append("Prediksi A (hampir sempurna)")

    # Prediksi B (under-segmentation)
    p2 = np.zeros_like(img, dtype=bool)
    cv2.circle(p2.view(np.uint8), (128, 128), 35, 255, -1)
    prediksi_list.append(p2.astype(bool))
    label_pred.append("Prediksi B (under-segmentation)")

    # Prediksi C (over-segmentation)
    p3 = np.zeros_like(img, dtype=bool)
    cv2.circle(p3.view(np.uint8), (128, 128), 65, 255, -1)
    prediksi_list.append(p3.astype(bool))
    label_pred.append("Prediksi C (over-segmentation)")

    # Prediksi D (salah posisi)
    p4 = np.zeros_like(img, dtype=bool)
    cv2.circle(p4.view(np.uint8), (80, 80), 30, 255, -1)
    prediksi_list.append(p4.astype(bool))
    label_pred.append("Prediksi D (posisi salah)")

    print(f"\n  {'Prediksi':<35} {'PA':>6} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Recall':>6}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for pred, lbl in zip(prediksi_list, label_pred):
        pa_val   = pixel_accuracy(pred, gt)
        iou_val  = iou(pred, gt)
        dice_val = dice(pred, gt)
        prec_val, rec_val = precision_recall(pred, gt)

        print(f"  {lbl:<35} {pa_val:>6.3f} {iou_val:>6.3f} {dice_val:>6.3f} {prec_val:>6.3f} {rec_val:>6.3f}")

    print("\n  Catatan: IoU dan Dice berkorelasi -> Dice = 2*IoU / (1 + IoU)")

    # Visualisasi
    gambar_vis = [img]
    judul_vis  = ["Citra Asli"]

    def buat_overlay_evaluasi(citra, pred, gt):
        """Hijau=TP, Merah=FP, Biru=FN"""
        vis = cv2.cvtColor(citra, cv2.COLOR_GRAY2RGB).copy()
        vis[np.logical_and(pred,  gt)]  = [80, 200, 80]   # TP
        vis[np.logical_and(pred, ~gt)]  = [200, 80, 80]   # FP
        vis[np.logical_and(~pred,  gt)] = [80, 80, 200]   # FN
        return vis

    for pred, lbl in zip(prediksi_list, label_pred):
        iou_v  = iou(pred, gt)
        dice_v = dice(pred, gt)
        overlay = buat_overlay_evaluasi(img, pred, gt)

        gambar_vis.append(overlay)
        judul_vis.append(f"{lbl}\nIoU={iou_v:.3f} | Dice={dice_v:.3f}")

    tampilkan_hasil(
        "Evaluasi Segmentasi (Hijau=TP, Merah=FP, Biru=FN)",
        gambar_vis,
        judul_vis,
        cmap_list=['gray'] + [None]*4,
        simpan=path_output('out_6_evaluasi.png')
    )


if __name__ == "__main__":
    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)
    demo_evaluasi(img)