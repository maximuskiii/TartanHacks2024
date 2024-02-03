import make_graph
import get_map
import requests
import googlemaps
import numpy as np
import cv2
import requests
from PIL import Image
import random
from io import BytesIO


api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo" 
latitude = 40.443518 
longitude = -79.942947  
zoom = 16 
size = "800x400"  
save_path = "/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/test.jpg" 

#get_map.save_custom_map_image(api_key, latitude, longitude, zoom, size, "roadmap", save_path)
#make_graph.id_paths('Maps+Paths/test.jpg', 'arr.npy', 50)

#def cumulative_test(address): 
#geocoding = get_map.addr_to_coord(address)
    



def getMapImg(api_key, latitude, longitude, zoom, size="600x300", maptype="roadmap", save_path="path/to/your/folder/map_image.jpg"):

    location = f"{latitude},{longitude}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        res = response.content
        #with open(save_path, 'wb') as file:
        #    file.write(response.content)
        #print(f"Map image saved to {save_path}")
    else:
        print("Error fetching the map image")\
    
    return res



def process_map_image(api_key, address, zoom, bar_height, size="600x300", maptype="roadmap"):

    geocoding = get_map.addr_to_coord(address)
    location = f"{geocoding[0]},{geocoding[1]}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image_array_rgb = np.array(image)
        img = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    else:
        print("Error, no map")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_img = np.zeros_like(img)
    cv2.drawContours(black_img, contours, -1, (255, 255, 255), 4)
    
    height, width = black_img.shape[:2]
    new_height = height - bar_height
    cropped_black_img = black_img[0:new_height, 0:width]
    
    #cv2.imshow('Paths Identified', cropped_black_img)
    #cv2.waitKey(0)  # Wait for a key press to close the image window
    #cv2.destroyAllWindows()

    return np.array(cropped_black_img)




def assign_random_color_to_white_pixels(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.all(img[i, j] == 255):
                random_color = np.array([random.randint(0, 255) for _ in range(3)])
                img[i, j] = random_color
    return img



def apply_heatmap_gradient(img):
    height = img.shape[0]
    
    def get_heatmap_color(y, height):
        normalized_position = y / height
        color = np.array([255 * normalized_position, 0, 255 * (1 - normalized_position)], dtype=np.uint8)
        return color

    for i in range(height): 
        for j in range(img.shape[1]):
            if np.all(img[i, j] == 255):
                heatmap_color = get_heatmap_color(i, height)
                img[i, j] = heatmap_color
    return img

#img = process_map_image(api_key, "5000 Forbes Ave, Pittsburgh, PA 15213", 16, 0, "600x300", "roadmap")
#cv2.imshow("testing", apply_heatmap_gradient(img))
#cv2.waitKey(0)
#cv2.destroyAllWindows()


import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def create_randomized_heatmap(height, width):
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return img

def generate_heatmap_frames(img, n_frames=10):
    frames_dir = 'heatmap_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    frame_filenames = []

    step_size = img.shape[0] // n_frames
    
    for i in range(n_frames):
        frame = np.zeros_like(img)
        reveal_until_row = step_size * (i + 1)
        frame[:reveal_until_row, :, :] = img[:reveal_until_row, :, :]
        
        frame_filename = f'{frames_dir}/frame_{i:02d}.png'
        plt.imsave(frame_filename, frame)
        frame_filenames.append(frame_filename)
    
    return frame_filenames

def create_animation(frame_filenames, output_filename='heatmap_animation.gif'):
    with imageio.get_writer(output_filename, mode='I') as writer:
        for filename in frame_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for filename in frame_filenames:
            os.remove(filename)


height, width = 200, 200  
img = create_randomized_heatmap(height, width)

frame_filenames = generate_heatmap_frames(img, n_frames=20)

create_animation(frame_filenames, 'heatmap_animation.gif')



