from PIL import Image
import math

def stitch_images(image_paths, images_per_row):
    """
    Stitch together multiple images into a super high-resolution image.
    
    Parameters:
    - image_paths: List of paths to the images to be stitched.
    - images_per_row: How many images should be placed in each row.
    
    Returns:
    - A PIL Image object representing the stitched high-resolution image.
    """
    # Open the first image to get its size
    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size
    
    # Calculate the grid size
    num_images = len(image_paths)
    num_rows = math.ceil(num_images / images_per_row)
    total_width = img_width * images_per_row
    total_height = img_height * num_rows
    
    # Create a new image with the calculated size
    super_img = Image.new('RGB', (total_width, total_height))
    
    # Paste each image into the correct position
    for index, img_path in enumerate(image_paths):
        with Image.open(img_path) as img:
            # Calculate the position
            x_offset = (index % images_per_row) * img_width
            y_offset = (index // images_per_row) * img_height
            super_img.paste(img, (x_offset, y_offset))
    
    return super_img