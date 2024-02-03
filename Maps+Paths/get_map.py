import requests

def save_custom_map_image(api_key, latitude, longitude, zoom, size="600x300", maptype="roadmap", save_path="path/to/your/folder/map_image.jpg"):
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
        # Save the image to a file
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Map image saved to {save_path}")
    else:
        print("Error fetching the map image")

# Example usage
api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"  # Replace with your actual API key
latitude = 40.443391 # Example: Latitude for Paris, France 40.443518, -79.942947
longitude = -79.942994  # Example: Longitude for Paris, France
zoom = 15 # Example zoom level
size = "800x400"  # Example size, width x height in pixels
save_path = "/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/test.jpg"  # Specify your desired save path

save_custom_map_image(api_key, latitude, longitude, zoom, size, "roadmap", save_path)
