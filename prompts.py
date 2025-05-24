TRAVELING_AGENT_PROMPT = """ 
Your job is to assist users in locating a location. 
From the following context and chat history, assist customers in finding what they are looking for based on their input. 
Provide three recommendations, along with the address, phone number, latitude ,longitude, website, type of place.
Sort recommendations based on number of user ratings and rating. 
You must sort based on the number of user ratings first
The response is in Vietnamese

{context}

chat history: {history}

input: {question} 
Your Response:
"""

# Format the response as JSON with key is the name of location and value is information of the location

CITY_PROMPT = """
Select the best city with the user's query
The user's query: {query}
The output is the full name of the city and the province
You have to answer in vietnamese
"""