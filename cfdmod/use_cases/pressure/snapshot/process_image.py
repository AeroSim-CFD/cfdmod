from PIL import Image


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
    top = (original_height - crop_height) / 2
    right = (original_width + crop_width) / 2
    bottom = (original_height + crop_height) / 2

    cropped_image = original_image.crop((left, top, right, bottom))

    return cropped_image


def paste_watermark(main_image: Image, watermark_image: Image):
    watermark = Image.open("./output/snapshot/axis_icon.png")
    main_image.paste(
        watermark,
        (
            int((main_image.width - watermark.width) / 2),
            int((main_image.height - 2 * watermark.height) / 2),
        ),
        watermark,
    )
