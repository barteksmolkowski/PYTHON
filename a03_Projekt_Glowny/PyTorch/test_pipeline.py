import matplotlib.pyplot as plt
import numpy as np

from preprocessing.pipeline import TransformPipeline


def interactive_inspection():
    pipeline = TransformPipeline()
    total_reviewed = 0
    bad_samples = []

    print("\n" + "=" * 50)
    print(" INTERACTIVE DATA INSPECTOR 2026 (100-Square Version) ")
    print("=" * 50)
    print("CONTROL:")
    print(" [Enter] - Next full batch (100 images)")
    print(" [D]     - Report an error (provide IDs of corrupted images)")
    print(" [Q]     - Finish and show report")
    print("=" * 50)

    while True:
        display_batch = []
        base_img = np.zeros((500, 500, 3), dtype=np.uint8)
        base_img[150:151, 150:400] = 255
        base_img[350:351, 150:400] = 255
        base_img[150:400, 350:351] = 255
        base_img[150:400, 150:151] = 255
        print(base_img)


        while len(display_batch) < 100:
            new_samples = pipeline.apply(base_img)
            print(f"nowe")
            display_batch.extend(new_samples)

        display_batch = display_batch[:100]

        fig, axes = plt.subplots(10, 10, figsize=(18, 18))
        current_batch_no = total_reviewed // 100 + 1
        fig.canvas.manager.set_window_title(
            f"Batch No. {current_batch_no} (100 samples)"
        )

        axes_flat = axes.flatten()

        for i in range(100):
            mtx = display_batch[i]
            d_img = (mtx - mtx.min()) / (mtx.max() - mtx.min() + 1e-8)

            axes_flat[i].imshow(d_img, cmap="gray")
            axes_flat[i].set_title(f"ID:{i}", fontsize=8)
            axes_flat[i].axis("off")

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show(block=False)
        plt.pause(0.1)

        cmd = (
            input(f"\n[Batch {current_batch_no}] Command (Enter/D/Q): ").strip().upper()
        )

        if cmd == "Q":
            plt.close("all")
            print(f"\n[FINAL REPORT]")
            print(f"Total samples reviewed: {total_reviewed}")
            if bad_samples:
                print("DETECTED ISSUES:")
                for bug in bad_samples:
                    print(f"  - {bug}")
            else:
                print("STATUS: All data looks GOOD (GIT).")
            break

        elif cmd == "D":
            wrong_ids = input("Enter IDs of corrupted images (0-99): ")
            bad_samples.append(f"Batch {current_batch_no}: IDs[{wrong_ids}]")
            print(f"Errors logged for IDs: {wrong_ids}. Moving forward...")

        plt.close(fig)
        total_reviewed += 100


if __name__ == "__main__":
    interactive_inspection()
