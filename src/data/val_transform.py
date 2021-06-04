import albumentations as A
import albumentations.pytorch as AP
import cv2


class ValidationTransform(A.Compose):
    """Image transformer for validation dataset.

    Since images in `BMS Molecular Translation` have different resolutions, this class
    resizes and pads to match the image size to `384x384` with preserving aspect ratio.

    Args:
        image_size: The desired input image resolution. Default is `384`.
    """

    def __init__(self, image_size: int = 384):
        super().__init__(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=0.5, std=0.5),
                AP.ToTensorV2(),
            ]
        )
