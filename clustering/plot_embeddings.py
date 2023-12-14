
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from PIL import Image

from utils import normalize_embeddings, load_embeddings, load_image

def plot_embeddings(embedding_type, save_filename=None, use_images=False):
    embeddings, labels, paths = load_embeddings(embedding_type)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    embeddings = normalize_embeddings(embeddings)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    if (labels.ndim > 1):
        label_colors = np.argmax(labels, axis=1)
    else:
        label_colors = labels
    
    plt.rcParams["image.cmap"] = "Spectral"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if not use_images:
        scatter = ax.scatter(output[:, 0], output[:, 1], c=label_colors)
        legend1 = ax.legend(*scatter.legend_elements(num=None), loc="upper right", title="Classes")
        ax.add_artist(legend1)
    else:
        tx, ty = output[:, 0], output[:, 1]
        tx, ty = (tx - np.min(tx)) / (np.max(tx) - np.min(tx)), (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        max_dim = 100
        width, height = 2048, 2048
        full_image = Image.new('RGBA', (width, height))
        from tqdm import tqdm
        for i in tqdm(range(output.shape[0])):
            x, y = tx[i], ty[i]
            img = load_image(paths[i], normalize=False).permute(1, 2, 0).detach().cpu().numpy()
            tile = Image.fromarray((img * 255).astype(np.uint8)).resize((224, 224))
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.Resampling.LANCZOS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        plt.tick_params(left=False, labelleft=False) #remove ticks
        plt.tick_params(bottom=False, labelbottom=False) #remove ticks
        plt.box(False) #remove box
        
        ax.imshow(full_image)

    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    parser.add_argument("-s", "--save_filename", type=str, help="filename to save plot to")
    parser.add_argument("-i", "--use_images", action="store_true", help="use images instead of points")
    args = parser.parse_args()
    save_filename = args.save_filename
    if save_filename == "":
        save_filename = args.embedding_type

    plot_embeddings(args.embedding_type, save_filename=save_filename, use_images=args.use_images)