import requests
from requests.structures import CaseInsensitiveDict
import pandas as pd
import json
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter

api_key_geoapify = "ec8294c519294767ba0c6861434c3d88"
url_place = "https://api.geoapify.com/v1/geocode/search"
url_search = "https://www.searchapi.io/api/v1/search"

headers = CaseInsensitiveDict()
headers["Accept"] = "application/json"
params = {
	"text": "Thành phố Lạng Sơn",
	"country": "Việt Nam",
	"limit": 1,
	"lang": "vi",
	"apiKey": api_key_geoapify
}

# res = requests.request("GET", url=url_place, params=params, headers=headers)
# dt_destination = res.json()["features"][0]["properties"]

# # Create the circle
# api_key_searchapi = "dZ3hmBQcvs1L4yDVbCjL8aXt"
# radius = 3000
# circle_center = {"latitude": dt_destination["lat"], "longitude": dt_destination["lon"]}
# circle_radius = radius

#-----------hotel------------------
# params_hotel = {
# 	"engine": "google_maps",
# 	"q": "Hotels",
# 	"ll": f"@{circle_center['latitude']},{circle_center['longitude']},{radius}m",
# 	"hl": "vi",
# 	"api_key": api_key_searchapi
# }
# result_hotel = requests.request("GET", url=url_search, params=params_hotel, headers=headers).json()
# with open("./data_test/hotel.json", "w") as f:
# 	json.dump(result_hotel, f, indent=4)

with open("./data_test/hotel.json") as f:
	result_hotel = json.load(f)
df_hotel = pd.json_normalize(result_hotel['local_results'])
df_hotel["type"] = "Hotel"
print(df_hotel)
#////////////////////////////////////

#----------restaurant--------------------
# params_restaurant = {
# 	"engine": "google_maps",
# 	"q": "Restaurants",
# 	"ll": f"@{circle_center['latitude']},{circle_center['longitude']},{radius}m",
# 	"hl": "vi",
# 	"api_key": api_key_searchapi
# }
# result_restaurant = requests.request("GET", url=url_search, params=params_restaurant, headers=headers).json()
# with open("./data_test/restaurant.json", "w") as f:
# 	json.dump(result_restaurant, f, indent=4)

with open("./data_test/restaurant.json") as f:
	result_restaurant = json.load(f)
df_restaurant = pd.json_normalize(result_restaurant['local_results'])
df_restaurant["type"] = "Restaurant"
print(df_restaurant)
#///////////////////////////////////////

#----------tourist---------------
# params_tourist = {
# 	"engine": "google_maps",
# 	"q": "Tourist Attractions",
# 	"ll": f"@{circle_center['latitude']},{circle_center['longitude']},{radius}m",
# 	"hl": "vi",
# 	"api_key": api_key_searchapi
# }
# result_tourist = requests.request("GET", url=url_search, params=params_tourist, headers=headers).json()
# with open("./data_test/tourist.json", "w") as f:
# 	json.dump(result_tourist, f, indent=4)

with open("./data_test/tourist.json") as f:
	result_tourist = json.load(f)
df_tourist = pd.json_normalize(result_tourist['local_results'])
df_tourist["type"] = "Tourist"
print(df_tourist)
#//////////////////////////////

def clean_text(x):
    if isinstance(x, str):
        return x.replace('\xa0', ' ')
    elif isinstance(x, list):
        return [item.replace('\xa0', ' ') if isinstance(item, str) else item for item in x]
    return x

df_place = pd.concat([df_hotel, df_restaurant, df_tourist], ignore_index=True)
print(df_place)
df_place = df_place.sort_values(by=['reviews', 'rating'], ascending=[False, False]).reset_index(drop=True)
df_place = df_place.applymap(clean_text)
df_place['combined_info'] = df_place.apply(lambda row: f"Type: {row['type']}, Name: {row['title']}. Rating: {row['rating']}. Address: {row['address']}. Website: {row['website']}", axis=1)
df_place_rename = df_place.rename(columns={
	'type': 'Type',
    'title': 'Name',
    'address': 'Address',
    'phone': 'Phone',
    'rating': 'Rating',
    'reviews': 'User Rating Count',
    'reviews_link': 'Google Maps URL',
    'website': 'Website URL',
    'gps_coordinates.latitude': 'Latitude',
    'gps_coordinates.longitude': 'Longitude',
    'thumbnail': 'Thumbnail'
})
df_place_brief = df_place_rename[['Type', 'Name', 'Address', 'Phone', 'Rating', 'User Rating Count','Google Maps URL', 'Website URL', 'Latitude', 'Longitude', 'Thumbnail']]

# Load Processed Dataset
loader = DataFrameLoader(df_place_rename, page_content_column="combined_info")
docs  = loader.load()
print(docs[0])
