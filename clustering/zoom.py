import numpy as np
from utils import load_image_big, save_img
import matplotlib.pyplot as plt

def run():
    masks_npz = np.load("../embeddings/masks.npz")
    masks, labels, paths = masks_npz["masks"], masks_npz["labels"], masks_npz["paths"]
    for (mask, label, path) in zip(masks, labels, paths):
        path_split = path.split("/")
        path_split[1] = "data_zoom"
        path = "/".join(path_split)
        img = load_image_big(path, normalize=False).permute((1, 2, 0)).detach().cpu().numpy()
        continue
        mask = np.reshape(mask, (16, 16))
        low_y = 16
        high_y = 0
        low_x = 16
        high_x = 0
        for i in range(16):
            for j in range(16):
                if mask[i][j]:
                    low_y = min(low_y, i)
                    high_y = max(high_y, i)
                    low_x = min(low_x, j)
                    high_x = max(high_x, j)

        if low_y == 16:
            low_y = 0
            high_y = 15
        if low_x == 16:
            low_x = 0
            high_x = 15

        if high_y - low_y > high_x - low_x:
            l_adjust = (high_y - low_y - high_x + low_x) / 2
            r_adjust = (high_y - low_y - high_x + low_x) / 2
            if low_x - l_adjust < 0:
                r_adjust += l_adjust - low_x
                l_adjust = low_x
            if high_x + r_adjust > 15:
                l_adjust += high_x + r_adjust - 15
                r_adjust = 15
            low_x -= l_adjust
            high_x += r_adjust

        if high_x - low_x > high_y - low_y:
            l_adjust = (high_x - low_x - high_y + low_y) / 2
            r_adjust = (high_x - low_x - high_y + low_y) / 2
            if low_y - l_adjust < 0:
                r_adjust += l_adjust - low_y
                l_adjust = low_y
            if high_y + r_adjust > 15:
                l_adjust += high_y + r_adjust - 15
                r_adjust = 15
            low_y -= l_adjust
            high_y += r_adjust

        high_x += 1
        high_y += 1

        img = load_image_big(path, normalize=False).permute((1, 2, 0)).detach().cpu().numpy()
        h, w = img.shape[:2]
        img_cropped = img[int(low_y * h / 16):int(high_y * h / 16 - 1), int(low_x * w / 16):int(high_x * w / 16 - 1)]

        path_split = path.split("/")
        path_split[1] = "data_zoom"
        path = "/".join(path_split)
        save_img(img_cropped, path)

        # fig = plt.figure()

        # fig.add_subplot(1, 2, 1)
        # plt.imshow((img * 255).astype(np.uint8))

        # fig.add_subplot(1, 2, 2)
        # plt.imshow((img_cropped * 255).astype(np.uint8))

        # plt.show()

        

if __name__ == "__main__":
    run()
