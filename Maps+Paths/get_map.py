import requests
import googlemaps

gmaps =  googlemaps.Client("AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo")

def addr_to_coord(addr): 
    geocoding = gmaps.geocode(addr)
    if geocoding:
        lat = geocoding[0]['geometry']['location']['lat']
        lng = geocoding[0]['geometry']['location']['lng']
        print(lat, lng)
    return (lat, lng)


def getMapImg(api_key, latitude, longitude, zoom, size="1200x600", maptype="roadmap", save_path="path/to/your/folder/map_image.jpg"):
    location = f"{latitude},{longitude}"
    style = "feature:all|element:labels|visibility:off"
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

    response = requests.get(map_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Map image saved to {save_path}")
    else:
        print("Error fetching the map image")


api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"  
latitude = 40.443391 
longitude = -79.942994  
zoom = 16 
size = "1200x600"  
save_path = "/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/test.jpg"
addr_coord = addr_to_coord("5000 Forbes Ave, Pittsburgh, PA 15213") 

getMapImg(api_key, latitude, longitude,zoom, size, "roadmap", save_path)