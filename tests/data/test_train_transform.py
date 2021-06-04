import numpy as np

from data.train_transform import TrainTransform


def test_TrainTransform_output_shape():
    transform = TrainTransform(image_size=384)

    img = np.random.randint(0, 0xFF, (245, 621, 3), dtype=np.uint8)
    assert transform(image=img)["image"].shape == (3, 384, 384)

    img = np.random.randint(0, 0xFF, (210, 247, 3), dtype=np.uint8)
    assert transform(image=img)["image"].shape == (3, 384, 384)

    img = np.random.randint(0, 0xFF, (11, 254, 1), dtype=np.uint8)
    assert transform(image=img)["image"].shape == (1, 384, 384)

    img = np.random.randint(0, 0xFF, (384, 512, 1), dtype=np.uint8)
    assert transform(image=img)["image"].shape == (1, 384, 384)

    img = np.random.randint(0, 0xFF, (384, 384, 1), dtype=np.uint8)
    assert transform(image=img)["image"].shape == (1, 384, 384)
