import requests
import googlemaps
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO

api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"
gmaps =  googlemaps.Client("AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo")

def addr_to_coord(addr): 
    geocoding = gmaps.geocode(addr)
    if geocoding:
        lat = geocoding[0]['geometry']['location']['lat']
        lng = geocoding[0]['geometry']['location']['lng']
        print(lat, lng)
    return (lat, lng)

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



    geocoding = addr_to_coord(address)
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

process_map_image(api_key, "5000 Forbes Ave, Pittsburgh, PA 15213", 16, 30, "600x300", "roadmap")
