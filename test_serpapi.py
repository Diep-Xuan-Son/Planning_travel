import requests


url_destination = "https://serpapi.com/locations.json"

headers = {
	'Content-Type': 'application/json',
}
params = {
	"q": "Hanoi",
	"limit": 1
}

res = requests.request("GET", url=url_destination, params=params, headers=headers)
print(res.json())
dt_destination = res.json()[0]

# Create the circle
radius = 3000
circle_center = {"latitude": dt_destination["gps"][1], "longitude": dt_destination["gps"][0]}
circle_radius = radius

url_search = "https://serpapi.com/search.json"
params = {
	"engine": "google_local",
	"q": "Hotel",
	"ll": f"@{circle_center['latitude']},{circle_center['longitude']},14z",
}

res = requests.request("GET", url=url_search, params=params, headers=headers)
print(res.json())
