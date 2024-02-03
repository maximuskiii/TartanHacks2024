from PIL import Image
import math

def stitch_images(image_paths, images_per_row):

    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size
    
    num_images = len(image_paths)
    num_rows = math.ceil(num_images / images_per_row)
    total_width = img_width * images_per_row
    total_height = img_height * num_rows
    
    super_img = Image.new('RGB', (total_width, total_height))

    for index, img_path in enumerate(image_paths):
        with Image.open(img_path) as img:
            x_offset = (index % images_per_row) * img_width
            y_offset = (index // images_per_row) * img_height
            super_img.paste(img, (x_offset, y_offset))
    
    return super_img