import requests
from io import BytesIO
import matplotlib.pyplot as plt

def fetch_and_display_map(api_key, location, zoom, size="600x300", maptype="roadmap"):
    """
    Fetches a map image from the Google Maps Static API and displays it.

    Parameters:
    - api_key: Your Google Maps API key as a string.
    - location: The center of the map (latitude,longitude) as a string.
    - zoom: The zoom level of the map as an integer.
    - size: The size of the map image in pixels (widthxheight) as a string. Default is "600x300".
    - maptype: The type of map to construct. Options include roadmap, satellite, hybrid, and terrain.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {
        "center": location,
        "zoom": zoom,
        "size": size,
        "maptype": maptype,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Use BytesIO to convert the bytes response into a binary stream
        image = BytesIO(response.content)
        # Load this image into matplotlib
        img = plt.imread(image, format='jpg')
        plt.imshow(img)
        plt.axis('off')  # No axis for a cleaner look
        plt.show()
    else:
        print("Error fetching the map image")

# Example usage
api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"  # Replace with your actual API key
location = "40.714728,-73.998672"  # Example location (New York City)
zoom = 12  # Example zoom level

fetch_and_display_map(api_key, location, zoom)

