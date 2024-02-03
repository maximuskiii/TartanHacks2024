import requests
import googlemaps
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
import io
import random
gmaps =  googlemaps.Client("YOUR_API_KEY")

#########################################
#####For Julius' drone path planning#####
#########################################


def addr_to_coord(addr): 
    geocoding = gmaps.geocode(addr)
    if geocoding:
        lat = geocoding[0]['geometry']['location']['lat']
        lng = geocoding[0]['geometry']['location']['lng']
        print(lat, lng)
    return (lat, lng)

def process_map_image(api_key, address, zoom, bar_height, size="600x300", maptype="roadmap"):


    geocoding = addr_to_coord(address)
    location = f"{geocoding[0]},{geocoding[1]}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image_array_rgb = np.array(image)
        img = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    else:
        print("Error fetching the map image")
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
    
    cv2.imshow('Paths Identified', cropped_black_img)
    print(cropped_black_img == np.array(cropped_black_img))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    return np.array(cropped_black_img)
    # np.save(npy_array_path, cropped_black_img)  # You need to define npy_array_path

#process_map_image(api_key, "5000 Forbes Ave, Pittsburgh, PA 15213", 16, 30, "600x300", "roadmap")

#####################################
#### For Alon - Satellite Image #####
#####################################


def getSatImg(address, zoom, api_key, size="1200x600", maptype="satellite", save_path="YOUR_FILE_PATH"):
    geocoding = addr_to_coord(address)
    location = f"{geocoding[0]},{geocoding[1]}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Map image saved to {save_path}")
    else:
        print("Error fetching the map image")
        
#getSatImg("Carnegie Mellon University", 17)

def getSatImgAsArray(address, zoom, api_key, size="1200x600", maptype="satellite"):
    geocoding = addr_to_coord(address) 
    location = f"{geocoding[0]},{geocoding[1]}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:

        image = Image.open(BytesIO(response.content))
        image_array_rgb = np.array(image)
        #image = Image.open(io.BytesIO(response.content))
        #image_array = np.array(image)
        return image_array_rgb
    else:
        print("Error fetching the map image")

#print(getSatImgAsArray("Carnegie Mellon University", 17))
        

####################################################
### For Alon and Ben - 1/0's path representation ###
####################################################       

def map_to_bin_graph(address, zoom, api_key, bar_height=0,  size="600x300", maptype="roadmap"):

    geocoding = addr_to_coord(address)
    location = f"{geocoding[0]},{geocoding[1]}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image_array_rgb = np.array(image)
        img = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    else:
        print("Error fetching the map image")
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
    #print(cropped_black_img == np.array(cropped_black_img))
    #cv2.waitKey(0)  # Wait for a key press to close the image window
    #cv2.destroyAllWindows()

    arr_img = np.array(cropped_black_img)
    bin_arr_img = np.where(arr_img == 255, 1, 0)
    print(bin_arr_img)
    return bin_arr_img

map_to_bin_graph("Carnegie Mellon University", 17)
