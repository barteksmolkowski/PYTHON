import cv2
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.pipeline import TransformPipeline


def interactive_inspection():
    pipeline = TransformPipeline()
    total_reviewed = 0
    bad_samples = []
    bad_images_data = []

    print("\n" + "=" * 60)
    print(" INTERACTIVE DATA INSPECTOR 2026 (Visual Debug Edition) ")
    print("=" * 60)

    while True:

        display_batch = []
        base_img = np.zeros((100, 100), dtype=np.uint8)

        cv2.putText(base_img, "5", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)

        while len(display_batch) < 100:
            new_samples = pipeline.apply(base_img)
            display_batch.extend(new_samples)

        display_batch = display_batch[:100]

        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        current_batch_no = (total_reviewed // 100) + 1
        fig.canvas.manager.set_window_title(f"Batch No. {current_batch_no}")

        axes_flat = axes.flatten()
        for i in range(100):
            mtx = display_batch[i]

            m_min, m_max = mtx.min(), mtx.max()
            d_img = (mtx - m_min) / (m_max - m_min + 1e-8)
            if np.mean(d_img) > 0.5:
                d_img = 1.0 - d_img

            axes_flat[i].imshow(d_img, cmap="gray")
            axes_flat[i].set_title(f"ID:{i}", fontsize=7)
            axes_flat[i].axis("off")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.2)

        cmd = (
            input(f"\n[Batch {current_batch_no}] Command (Enter/D/Q): ").strip().upper()
        )

        if cmd == "D":
            user_input = input("Enter IDs of corrupted images (comma-separated): ")
            try:
                ids = [int(x.strip()) for x in user_input.split(",") if x.strip()]
                for idx in ids:
                    if 0 <= idx < 100:
                        bad_samples.append(f"Batch {current_batch_no}, ID: {idx}")
                        bad_images_data.append(display_batch[idx].copy())
                print(f"Logged {len(ids)} issues. Moving to next batch...")
            except ValueError:
                print("[!] Invalid IDs format. Skipping log...")

        elif cmd == "Q":
            plt.close("all")
            np.set_printoptions(threshold=np.inf, linewidth=200)

            print("\n" + "!" * 20 + " RAW MATRIX DUMP (FULL) " + "!" * 20)

            if bad_images_data:
                for i, mtx in enumerate(bad_images_data):
                    print(f"\n[ FULL DATA FOR {bad_samples[i]} ]")
                    print(
                        f"Stats: Min={np.min(mtx)}, Max={np.max(mtx)}, Mean={np.mean(mtx):.2f}"
                    )
                    print("-" * 30)
                    print(mtx)
                    print("-" * 30)

                print("\nNow you can copy the entire matrices for analysis.")
                np.set_printoptions(threshold=1000)
            else:
                print("STATUS: No corrupted samples to display.")
            break

        plt.close(fig)
        total_reviewed += 100


if __name__ == "__main__":
    interactive_inspection()
