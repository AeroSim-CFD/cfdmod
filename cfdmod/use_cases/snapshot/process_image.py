import pathlib

from PIL import Image

from cfdmod.use_cases.snapshot.config import CropConfig


def crop_image_center(original_image: Image, width_ratio: float, height_ratio: float) -> Image:
    """Crops a image based on the center

    Args:
        original_image (Image): Original image
        width_ratio (float): Width ratio for the crop
        height_ratio (float): Heigth ratio for the crop

    Returns:
        Image: Cropped image
    """
    original_width, original_height = original_image.size
    crop_width = original_width * width_ratio
    crop_height = original_height * height_ratio

    left = (original_width - crop_width) / 2
    right = (original_width + crop_width) / 2
    top = original_height - crop_height
    bottom = original_height

    cropped_image = original_image.crop((left, top, right, bottom))

    return cropped_image


def paste_watermark(main_image: Image, watermark_image: Image):
    """Adds a watermark to the main image

    Args:
        main_image (Image): Main Image
        watermark_image (Image): Watermark image
    """
    main_image.paste(
        watermark_image,
        (
            int((main_image.width - watermark_image.width) / 2),
            int((main_image.height - 2 * watermark_image.height) / 2),
        ),
        watermark_image,
    )


def process_image(image_path: pathlib.Path, output_path: pathlib.Path, crop_cfg: CropConfig):
    """Processes the generated image

    Args:
        image_path (pathlib.Path): Path of the generated image
        crop_cfg (CropConfig): Image post processing parameters

    Returns:
        Image: Processed image
    """
    image = Image.open(image_path)
    cropped_image = crop_image_center(
        original_image=image, width_ratio=crop_cfg.width_ratio, height_ratio=crop_cfg.height_ratio
    )
    if crop_cfg.watermark_path is not None:
        watermark = Image.open(crop_cfg.watermark_path)
        paste_watermark(main_image=cropped_image, watermark_image=watermark)

    cropped_image.save(output_path)
