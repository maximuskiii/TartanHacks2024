import make_graph
import get_map

api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"  # Replace with your actual API key
latitude = 40.443518  # Example: Latitude for Paris, France 40.443518, -79.942947
longitude = -79.942947  # Example: Longitude for Paris, France
zoom = 16  # Example zoom level
size = "800x400"  # Example size, width x height in pixels
save_path = "/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/test.jpg"  # Specify your desired save path

get_map.save_custom_map_image(api_key, latitude, longitude, zoom, size, "roadmap", save_path)
make_graph.id_paths('Maps+Paths/test.jpg', 'arr.npy', 50)