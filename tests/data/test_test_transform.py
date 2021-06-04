import numpy as np

from data.test_transform import TestTransform


def test_TestTransform_output_shape():
    transform = TestTransform(image_size=384)

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
