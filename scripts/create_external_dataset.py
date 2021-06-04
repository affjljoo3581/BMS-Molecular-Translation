import argparse
import math
import os
import signal
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
import tqdm


@contextmanager
def timeout(seconds: int):
    def _handle_timeout(*args: Any, **kwargs: Any):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.alarm(0)


def _create_mol_draw_options() -> Draw.MolDrawOptions:
    options = Draw.MolDrawOptions()
    options.useBWAtomPalette()

    options.additionalAtomLabelPadding = np.random.uniform(0.05, 0.3)
    options.bondLineWidth = 1
    options.multipleBondOffset = np.random.uniform(0.05, 0.2)

    options.rotate = np.random.uniform(0, 360)
    options.fixedScale = np.random.uniform(0.05, 0.07)
    options.maxFontSize = 40

    return options


def generate_image_from_inchi(inchi: str) -> np.ndarray:
    # Draw InChI molecular image.
    image_size = np.random.randint(300, 500)

    draw = Draw.rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
    draw.SetDrawOptions(_create_mol_draw_options())
    draw.DrawMolecule(Chem.MolFromInchi(inchi))
    draw.FinishDrawing()

    data = np.frombuffer(draw.GetDrawingText(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    # Crop paddings from the image and add small borders.
    img = img[~np.all(img == 0xFF, axis=1), :]
    img = img[:, ~np.all(img == 0xFF, axis=0)]

    # Randomly downscale the molecular images with bad interpolation algorithm (nearest)
    # to reduce the molecular quality.
    if np.random.rand() < 0.2:
        img = cv2.resize(img, (0, 0), fx=0.66, fy=0.66, interpolation=cv2.INTER_NEAREST)
        img = 0xFF * (img > 0x9F).astype(np.uint8)

    border = math.floor(np.random.uniform(0.1, 0.13) * max(img.shape))
    img = cv2.copyMakeBorder(
        img, border, border, border, border, cv2.BORDER_CONSTANT, value=0xFF
    )

    # Add random salt-and-pepper noise.
    noise = np.random.random(img.shape)
    img[noise < 0.00015] = 0
    img[noise > 0.85] = 0xFF
    return img


def main(args: argparse.Namespace):
    extra_inchis = pd.read_csv(args.extra_inchi_csv)

    # Split the inchis into the folds and choose one of them.
    fold_size = len(extra_inchis) // args.num_folds
    extra_inchis = extra_inchis.iloc[
        fold_size * args.fold_index : fold_size * (args.fold_index + 1)
    ]

    # Create an image id by using `hash` function.
    extra_inchis["image_id"] = extra_inchis["InChI"].map(
        lambda x: "{:012x}".format(0xFFFFFFFFFFFF & hash(x))
    )

    # Save the labels to `train_labels.csv` file.
    os.makedirs(args.output_path, exist_ok=True)
    extra_inchis[["image_id", "InChI"]].set_index("image_id").to_csv()

    # Render the inchi images and save to the output directory.
    with open(os.path.join(args.output_path, "train_labels.csv"), "w") as fp:
        fp.write("image_id,InChI\n")

        for sample in tqdm.tqdm(extra_inchis.itertuples(), total=len(extra_inchis)):
            base_dir = os.path.join(args.output_path, "train", *sample.image_id[:3])

            try:
                with timeout(10):
                    inchi_img = generate_image_from_inchi(sample.InChI)

                os.makedirs(base_dir, exist_ok=True)
                cv2.imwrite(os.path.join(base_dir, sample.image_id + ".png"), inchi_img)

                fp.write(f'{sample.image_id},"{sample.InChI}"\n')
            except TimeoutError:
                print(f"[*] skip {sample.image_id} due to timeout.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("extra_inchi_csv")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--num_folds", default=20, type=int)
    parser.add_argument("--fold_index", default=0, type=int)
    args = parser.parse_args()

    main(args)
