import requests
from requests.structures import CaseInsensitiveDict

api_key = "ec8294c519294767ba0c6861434c3d88"

url_place = "https://api.geoapify.com/v1/geocode/search"

headers = CaseInsensitiveDict()
headers["Accept"] = "application/json"

params = {
	"text": "Thành phố Lạng Sơn",
	"country": "Việt Nam",
	"limit": 1,
	"lang": "vi",
	"apiKey": api_key
}
res = requests.request("GET", url=url_place, params=params, headers=headers)
print(res.json())
dt_destination = res.json()["features"][0]["properties"]

#----------------------------------------
# Create the circle
radius = 3000
circle_center = {"latitude": dt_destination["lat"], "longitude": dt_destination["lon"]}
circle_radius = radius

# url_search = "https://api.geoapify.com/v2/places"
# params_hotel = {
# 	"categories": "accommodation.hotel",
# 	"filter": f"circle:{circle_center['longitude']},{circle_center['latitude']},{radius}",
# 	"bias": f"proximity:{circle_center['longitude']},{circle_center['latitude']}",
# 	"lang": "vi",
# 	"limit": 1,
# 	"apiKey": api_key
# }
# res = requests.request("GET", url=url_search, params=params_hotel, headers=headers)
# print(res.json())

# params_restaurant = {
# 	"categories": "accommodation.hotel",
# 	"filter": f"circle:{circle_center['longitude']},{circle_center['latitude']},{radius}",
# 	"bias": f"proximity:{circle_center['longitude']},{circle_center['latitude']}",
# 	"lang": "vi",
# 	"limit": 1,
# 	"apiKey": api_key
# }
# res = requests.request("GET", url=url_search, params=params_restaurant, headers=headers)
# print(res.json())

# params_tourist = {
# 	"categories": "accommodation.hotel",
# 	"filter": f"circle:{circle_center['longitude']},{circle_center['latitude']},{radius}",
# 	"bias": f"proximity:{circle_center['longitude']},{circle_center['latitude']}",
# 	"lang": "vi",
# 	"limit": 1,
# 	"apiKey": api_key
# }
# res = requests.request("GET", url=url_search, params=params_tourist, headers=headers)
# print(res.json())


#-----------------------------------------------
url_search_detail = "https://api.geoapify.com/v2/place-details"
params_hotel = {
	"lat": 21.8542927,
	"lon": 106.7568197,
	"features": f"details",
	"lang": "vi",
	"apiKey": api_key
}
res = requests.request("GET", url=url_search_detail, params=params_hotel, headers=headers)
print(res.json())
