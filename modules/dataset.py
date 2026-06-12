import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

from utils import generate_phoc_vector, generate_phos_vector


class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None, calc_phosc=True):
        """
        Builds self.df_all with columns: ["Image", "Word", "phos", "phoc", "phosc"]
        - phos : np.float32 (165,)
        - phoc : np.float32 (604,)
        - phosc: concat (165+604=769,)
        """
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(csvfile)

        # Ensure required columns
        if "Image" not in df.columns or "Word" not in df.columns:
            if df.shape[1] >= 2:
                df.columns = ["Image", "Word"] + [f"col_{i}" for i in range(df.shape[1] - 2)]
            else:
                raise ValueError("CSV must have at least two columns: Image, Word")

        df = df[["Image", "Word"]].copy()
        df["Word"] = df["Word"].astype(str)

        # Precompute vectors for unique words
        words = sorted(set(df["Word"].tolist()))
        phos_map = {w: np.asarray(generate_phos_vector(w), dtype=np.float32) for w in words}
        phoc_map = {w: np.asarray(generate_phoc_vector(w), dtype=np.float32) for w in words}

        df["phos"] = df["Word"].map(phos_map)
        df["phoc"] = df["Word"].map(phoc_map)
        df["phosc"] = [np.concatenate([p, q]).astype(np.float32) for p, q in zip(df["phos"], df["phoc"])]

        self.df_all = df

    def __getitem__(self, index):
        img_rel = self.df_all.iloc[index, 0]
        # support absolute or relative paths
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(self.root_dir, img_rel)
        image = io.imread(img_path)

        # Ensure RGB (model expects 3 channels)
        if image.ndim == 2:  # grayscale
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Target = concatenated PHOS+PHOC
        y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)



if __name__ == '__main__':
    from torchvision.transforms import transforms

    dataset = phosc_dataset('image_data/IAM_test_unseen.csv', '../image_data/IAM_test', transform=transforms.ToTensor())

    print(dataset.df_all)

    print(dataset.__getitem__(0))
