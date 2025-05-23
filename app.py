import os
import jwt
import folium
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from streamlit_folium import folium_static

from traveling_agent import TravelAgentTool

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
secret = "my sim chatbot"
api_key_openai_encode = os.getenv('API_KEY_OPENAI_ENCODE')
api_key_openai = jwt.decode(api_key_openai_encode, secret, algorithms=["HS256"])["api_key"]
api_key_searchapi = os.getenv('API_KEY_SEARCHAPI')

# TRAVEL_AGENT = TravelAgentTool(api_key=api_key_openai, api_key_searchapi=api_key_searchapi, dbmem_name="Memory", redis_url="", redis_port=6400)

def main():
    st.sidebar.title("Travel Recommendation App Demo")

    # api_key_searchapi = st.sidebar.text_input("Enter Google Maps API key:",type="password")
    # api_key_openai = st.sidebar.text_input("Enter OpenAI API key:",type="password")
    TRAVEL_AGENT = TravelAgentTool(api_key=api_key_openai, api_key_searchapi=api_key_searchapi, dbmem_name="Memory", redis_url="", redis_port=6400)

    st.sidebar.write('Please fill in the fields below.')
    destination = st.sidebar.text_input('Destination:',key='destination_app')
    min_rating = st.sidebar.number_input('Minimum Rating:',value=4.0,min_value=0.5,max_value=4.5,step=0.5,key='minrating_app')
    radius = st.sidebar.number_input('Search Radius in meter:',value=3000,min_value=500,max_value=50000,step=100,key='radius_app')

    if destination:
        df_place = TRAVEL_AGENT.prepare_data(destination=destination, min_rating=min_rating, radius=radius)
        df_place_rename = df_place[0]

        def database():
            st.dataframe(df_place[1])

        def maps():
            st.header("🌏 Travel Recommendation App 🌏")

            places_type = st.radio('Looking for: ',["Hotels 🏨", "Restaurants 🍴","Tourist Attractions ⭐"])
            initial_location = list(df_place[2].values())
            type_colour = {'Hotel':'blue', 'Restaurant':'green', 'Tourist':'orange'}
            type_icon = {'Hotel':'home', 'Restaurant':'cutlery', 'Tourist':'star'}

            st.write(f"# Here are our recommendations for {places_type} near {destination} ")

            if places_type == 'Hotels 🏨': 
                df_hotel = df_place_rename[df_place_rename["Type"]=="Hotel"]
                with st.spinner("Just a moment..."):
                    for index,row in df_hotel.iterrows():
                        location = [row['Latitude'], row['Longitude']]
                        mymap  = folium.Map(location = initial_location, 
                                zoom_start=9, control_scale=True)
                        content = (str(row['Name']) + '<br>' + 
                                'Rating: '+ str(row['Rating']) + '<br>' + 
                                'Address: ' + str(row['Address']) + '<br>' + 
                                'Website: '  + str(row['Website_URL'])
                                )
                        iframe = folium.IFrame(content, width=300, height=125)
                        popup = folium.Popup(iframe, max_width=300)

                        icon_color = type_colour[row['Type']]
                        icon_type = type_icon[row['Type']]
                        icon = folium.Icon(color=icon_color, icon=icon_type)

                        # Use different icons for hotels, restaurants, and tourist attractions
                        folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                        st.write(f"## {index + 1}. {row['Name']}")
                        folium_static(mymap)
                        st.write(f"Rating: {row['Rating']}")
                        st.write(f"Address: {row['Address']}")
                        st.write(f"Website: {row['Website_URL']}")
                        st.write(f"More information: {row['Google_Maps_URL']}\n")
                            
            elif places_type == 'Restaurants 🍴': 
                df_restaurant = df_place_rename[df_place_rename["Type"]=="Restaurant"]
                with st.spinner("Just a moment..."):
                    for index,row in df_restaurant.iterrows():
                        location = [row['Latitude'], row['Longitude']]
                        mymap  = folium.Map(location = initial_location, 
                                zoom_start=9, control_scale=True)
                        content = (str(row['Name']) + '<br>' + 
                                'Rating: '+ str(row['Rating']) + '<br>' + 
                                'Address: ' + str(row['Address']) + '<br>' + 
                                'Website: '  + str(row['Website_URL'])
                                )
                        iframe = folium.IFrame(content, width=300, height=125)
                        popup = folium.Popup(iframe, max_width=300)

                        icon_color = type_colour[row['Type']]
                        icon_type = type_icon[row['Type']]
                        icon = folium.Icon(color=icon_color, icon=icon_type)

                        # Use different icons for hotels, restaurants, and tourist attractions
                        folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                        st.write(f"## {index + 1}. {row['Name']}")
                        folium_static(mymap)
                        st.write(f"Rating: {row['Rating']}")
                        st.write(f"Address: {row['Address']}")
                        st.write(f"Website: {row['Website_URL']}")
                        st.write(f"More information: {row['Google_Maps_URL']}\n")
            else:
                df_tourist = df_place_rename[df_place_rename["Type"]=="Tourist"]
                with st.spinner("Just a moment..."):
                    for index,row in df_tourist.iterrows():
                        location = [row['Latitude'], row['Longitude']]
                        mymap  = folium.Map(location = initial_location, 
                                zoom_start=9, control_scale=True)
                        content = (str(row['Name']) + '<br>' + 
                                'Rating: '+ str(row['Rating']) + '<br>' + 
                                'Address: ' + str(row['Address']) + '<br>' + 
                                'Website: '  + str(row['Website_URL'])
                                )
                        iframe = folium.IFrame(content, width=300, height=125)
                        popup = folium.Popup(iframe, max_width=300)

                        icon_color = type_colour[row['Type']]
                        icon_type = type_icon[row['Type']]
                        icon = folium.Icon(color=icon_color, icon=icon_type)

                        # Use different icons for hotels, restaurants, and tourist attractions
                        folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                        st.write(f"## {index + 1}. {row['Name']}")
                        folium_static(mymap)
                        st.write(f"Rating: {row['Rating']}")
                        st.write(f"Address: {row['Address']}")
                        st.write(f"Website: {row['Website_URL']}")
                        st.write(f"More information: {row['Google_Maps_URL']}\n")

        def chatbot():
            class Message(BaseModel):
                actor: str
                payload : str

            llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature=0) 

            USER = "user"
            ASSISTANT = "ai"
            MESSAGES = "messages"

            # def initialize_session_state():
            if MESSAGES not in st.session_state:
                st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]
            
            msg: Message
            for msg in st.session_state[MESSAGES]:
                st.chat_message(msg.actor).write(msg.payload)

            # Prompt
            query: str = st.chat_input("Enter a prompt here")

            vector_dt = TRAVEL_AGENT.prepare_vector_data(data=df_place_rename)
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vector_dt[0].as_retriever(),
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": vector_dt[1],
                    "memory": vector_dt[2]}
            )

            if query:
                st.session_state[MESSAGES].append(Message(actor=USER, payload=str(query)))
                st.chat_message(USER).write(query)

                with st.spinner("Please wait..."):
                    response: str = qa.run(query = query)
                    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
                    st.chat_message(ASSISTANT).write(response)
               # st.write("Chatbot")

        method = st.sidebar.radio(" ",["Search 🔎","ChatBot 🤖","Database 📑"], key="method_app")
        if method == "Search 🔎":
            maps()
        elif method == "ChatBot 🤖":
            chatbot()
        else:
            database()

if __name__ == '__main__':
    main()
    
# streamlit run app.py --server.port 8502 --server.address 0.0.0.0