import requests
from io import BytesIO
import matplotlib.pyplot as plt

def get_bike_route_polyline(api_key, origin, destination):
    """
    Fetches a polyline for a bike route from the Google Maps Directions API.

    Parameters:
    - api_key: Your Google Maps API key as a string.
    - origin: The start location as a string in the format "latitude,longitude".
    - destination: The end location as a string in the format "latitude,longitude".

    Returns:
    - A polyline as a string if successful, None otherwise.
    """
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": "bicycling",
        "key": api_key
    }
    response = requests.get(directions_url, params=params)
    if response.status_code == 200 and response.json()["status"] == "OK":
        # Extract the polyline from the first route
        routes = response.json()["routes"]
        polyline = routes[0]["overview_polyline"]["points"]
        return polyline
    else:
        print("Failed to fetch bike route")
        return None

def display_map_with_bike_path(api_key, polyline):
    """
    Displays a map with a bike path highlighted, using the Google Maps Static API.

    Parameters:
    - api_key: Your Google Maps API key as a string.
    - polyline: The polyline string that represents the bike route.
    """
    size = "600x300"
    path = f"color:black|weight:5|enc:{polyline}"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?size={size}&path={path}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        image = BytesIO(response.content)
        img = plt.imread(image, format='jpeg')
        plt.imshow(img)
        plt.axis('off')  # Hide axis for a cleaner look
        plt.show()
    else:
        print("Error fetching the map image")

# Example usage
api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"  # You must replace this with your actual Google Maps API key
origin = "40.714728,-73.998672"  # Example origin coordinates (New York City)
destination = "40.695813,-73.987557"  # Example destination coordinates (Brooklyn, NY)

polyline = get_bike_route_polyline(api_key, origin, destination)
if polyline:
    display_map_with_bike_path(api_key, polyline)
else:
    print("Could not obtain bike route polyline.")