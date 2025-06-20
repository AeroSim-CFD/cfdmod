import pathlib

from PIL import Image
from IPython.display import display


from cfdmod.use_cases.snapshot.config import CropConfig, OverlayImageConfig


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

def paste_overlay_image(
    main_image_path: pathlib.Path, 
    image_to_overlay_config: OverlayImageConfig,
    output_path: pathlib.Path|None=None,
):
    """Adds a watermark to the main image

    Args:
        main_image (Image): Main Image
        watermark_image (Image): Watermark image
    """
    if output_path is None:
        output_path = main_image_path
    main_image = Image.open(main_image_path)
    image_to_overlay_path = image_to_overlay_config.image_path
    image_to_overlay = Image.open(image_to_overlay_path)
    #scale
    scale = image_to_overlay_config.scale
    (width, height) = (image_to_overlay.width, image_to_overlay.height)
    image_to_overlay = image_to_overlay.resize((int(width*scale), int(height*scale)))
    #transparency
    image_to_overlay = image_to_overlay.convert("RGBA")
    r, g, b, a = image_to_overlay.split()
    transparency = image_to_overlay_config.transparency
    alpha = 255 * (1-transparency)  # 0 (transparent) to 255 (opaque)
    a = a.point(lambda p: alpha if p > 0 else 0) #don't impact points that are already transparent
    image_to_overlay = Image.merge("RGBA", (r, g, b, a))
    #rotation
    angle = image_to_overlay_config.angle
    image_to_overlay = image_to_overlay.rotate(angle)
    #overlaying
    position = image_to_overlay_config.position
    position = (int(position[0]), int(position[1]))
    main_image.paste(
        image_to_overlay,
        position,
        image_to_overlay,
    )
    main_image.save(output_path)



def crop_image(image_path: pathlib.Path, crop_cfg: CropConfig, output_path: pathlib.Path|None=None):
    """Processes the generated image

    Args:
        image_path (pathlib.Path): Path of the generated image
        crop_cfg (CropConfig): Image post processing parameters

    Returns:
        Image: Processed image
    """
    if output_path is None:
        output_path = image_path
    image = Image.open(image_path)
    cropped_image = crop_image_center(
        original_image=image, width_ratio=crop_cfg.width_ratio, height_ratio=crop_cfg.height_ratio
    )
    cropped_image.save(output_path)

def display_image(image_path: pathlib.Path):
    img = Image.open(image_path)
    display(img)
