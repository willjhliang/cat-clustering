
import argparse
from glob import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import load_image

def visualize_dataset(data_dir, cats):
    paths = {}
    if cats is None:
        cats = ["*"]
    for cat in cats:
        for extension in ["jpg", "jpeg", "png"]:
            for filename in glob(os.path.join(data_dir, f"{cat}/*.{extension}"), recursive=True):
                if cat not in paths.keys():
                    paths[cat] = []
                paths[cat].append(filename)
    for cat in paths:
        random.shuffle(paths[cat])

    i = 0
    while True:
        path = paths[cats[i % len(cats)]][i // len(cats)]
        img = load_image(path, normalize=False).permute(1, 2, 0).detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1)
        ax.imshow((img * 255).astype(np.uint8))
        ax.set_title(path)
        plt.show()

        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../data", help="path to data directory")
    parser.add_argument("-c", "--categories", type=str, nargs="+", help="categories to visualize")
    args = parser.parse_args()

    visualize_dataset(args.data_dir, args.categories)