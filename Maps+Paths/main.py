import make_graph
import get_map
import requests
import googlemaps
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO


api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"  # Replace with your actual API key
latitude = 40.443518  # Example: Latitude for Paris, France 40.443518, -79.942947
longitude = -79.942947  # Example: Longitude for Paris, France
zoom = 16  # Example zoom level
size = "800x400"  # Example size, width x height in pixels
save_path = "/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/test.jpg"  # Specify your desired save path

#get_map.save_custom_map_image(api_key, latitude, longitude, zoom, size, "roadmap", save_path)
#make_graph.id_paths('Maps+Paths/test.jpg', 'arr.npy', 50)

#def cumulative_test(address): 
#geocoding = get_map.addr_to_coord(address)
    



def getMapImg(api_key, latitude, longitude, zoom, size="600x300", maptype="roadmap", save_path="path/to/your/folder/map_image.jpg"):
    """
    Fetches a Google Maps static image with specified parameters and saves it to a file.

    Parameters:
    - api_key: Your Google Maps API key as a string.
    - latitude: Latitude for the center of the map as a float.
    - longitude: Longitude for the center of the map as a float.
    - zoom: The zoom level of the map as an integer.
    - size: The size of the map image in pixels (widthxheight) as a string. Example: "600x300".
    - maptype: The type of map to display (e.g., "roadmap", "satellite").
    - save_path: Full path where the image will be saved, including the file name and extension.
    """
    location = f"{latitude},{longitude}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        res = response.content
        # Save the image to a file
        #with open(save_path, 'wb') as file:
        #    file.write(response.content)
        #print(f"Map image saved to {save_path}")
    else:
        print("Error fetching the map image")\
    
    return res



def process_map_image(api_key, address, zoom, bar_height, size="600x300", maptype="roadmap"):
    """
    Fetches a Google Maps static image, processes it to identify paths, and crops it based on a given bar height.
    
    Parameters:
    - api_key: Your Google Maps API key.
    - latitude: Center latitude of the map.
    - longitude: Center longitude of the map.
    - zoom: Zoom level of the map.
    - size: Size of the map image (widthxheight).
    - maptype: Type of map ("roadmap", "satellite", etc.).
    - bar_height: Height of the bar to crop from the bottom of the image.
    """

    geocoding = get_map.addr_to_coord(address)
    location = f"{geocoding[0]},{geocoding[1]}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    # Fetch the map image
    response = requests.get(map_url)
    if response.status_code == 200:
        # Convert the image to a numpy array
        image = Image.open(BytesIO(response.content))
        image_array_rgb = np.array(image)
        img = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    else:
        print("Error fetching the map image")
        return
    
    # Image processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_img = np.zeros_like(img)
    cv2.drawContours(black_img, contours, -1, (255, 255, 255), 4)
    
    # Cropping
    height, width = black_img.shape[:2]
    new_height = height - bar_height
    cropped_black_img = black_img[0:new_height, 0:width]
    
    # Display the processed image (optional)
    cv2.imshow('Paths Identified', cropped_black_img)
    print(cropped_black_img == np.array(cropped_black_img))
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

    return np.array(cropped_black_img)
    
    # Optionally, save the processed image as a numpy array
    # np.save(npy_array_path, cropped_black_img)  # You need to define npy_array_path

process_map_image(api_key, "5000 Forbes Ave, Pittsburgh, PA 15213", 13, 30, "600x300", "roadmap")

