import os
import jwt
import torch
import folium
import re as regex
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from streamlit.components.v1 import html
from langchain.chains import RetrievalQA
from streamlit_folium import folium_static

from traveling_agent import TravelAgentTool

torch.classes.__path__ = []

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
secret = "MMVMM"
#api_key_openai_encode = os.getenv('API_KEY_OPENAI_ENCODE')
#api_key_openai = jwt.decode(api_key_openai_encode, secret, algorithms=["HS256"])["api_key"]
api_key_searchapi = os.getenv('API_KEY_SEARCHAPI')

# TRAVEL_AGENT = TravelAgentTool(api_key=api_key_openai, api_key_searchapi=api_key_searchapi, dbmem_name="Memory", redis_url="", redis_port=6400)

def main():
    st.sidebar.title("Traveling App Demo")

    # api_key_searchapi = st.sidebar.text_input("Enter Google Maps API key:",type="password")
    api_key_openai = st.sidebar.text_input("Enter OpenAI API key:",type="password")
    if api_key_openai:
        # TRAVEL_AGENT = TravelAgentTool(api_key=api_key_openai, api_key_searchapi=api_key_searchapi, dbmem_name="Memory", redis_url="", redis_port=6400)
        if "travel_agent" not in st.session_state:
            st.session_state["travel_agent"] = TravelAgentTool(api_key=api_key_openai, api_key_searchapi=api_key_searchapi, dbmem_name="Memory", redis_url="", redis_port=6400)
        TRAVEL_AGENT = st.session_state["travel_agent"]

        st.sidebar.write('Hãy điền đầy đủ thông tin dưới đây.')
        destination = st.sidebar.text_input('Địa điểm:',key='destination_app')
        min_rating = st.sidebar.number_input('Mức đánh giá thấp nhất:',value=4.0,min_value=0.5,max_value=4.5,step=0.5,key='minrating_app')
        radius = st.sidebar.number_input('Bán kính tìm kiếm:',value=3000,min_value=500,max_value=50000,step=100,key='radius_app')

        if destination:
            if "current_destination" not in st.session_state:
                st.session_state["current_destination"] = destination
                st.session_state["df_place"] = TRAVEL_AGENT.prepare_data(destination=destination, min_rating=min_rating, radius=radius)
                st.session_state["vector_dt"] = TRAVEL_AGENT.prepare_vector_data(data=st.session_state["df_place"][0])
            else:
                if st.session_state["current_destination"] != destination:
                    st.session_state["current_destination"] = destination
                    st.session_state["df_place"] = TRAVEL_AGENT.prepare_data(destination=destination, min_rating=min_rating, radius=radius)
                    st.session_state["vector_dt"] = TRAVEL_AGENT.prepare_vector_data(data=st.session_state["df_place"][0])

            # df_place = TRAVEL_AGENT.prepare_data(destination=destination, min_rating=min_rating, radius=radius)
            # vector_dt = TRAVEL_AGENT.prepare_vector_data(data=TRAVEL_AGENT.df_place[0])
            df_place = st.session_state["df_place"]
            df_place_rename = df_place[0]

            st.session_state["initial_location"] = list(df_place[2].values())
            st.session_state["type_colour"] = {"Khách sạn":'blue', "Nhà hàng":'green', "Điểm du lịch":'orange'}
            st.session_state["type_icon"] = {"Khách sạn":'home', "Nhà hàng":'cutlery', "Điểm du lịch":'star'}

            def database():
                st.dataframe(df_place[1])

            def maps():
                st.header("🌏 Traveling App 🌏")

                places_type = st.radio('Tìm kiếm thông tin: ',["Khách sạn 🏨", "Nhà hàng 🍴","Điểm du lịch hấp dẫn ⭐"])
                initial_location = st.session_state["initial_location"]
                type_colour = st.session_state["type_colour"]
                type_icon = st.session_state["type_icon"]

                st.write(f"# Dưới đây là một số đề xuất {places_type.lower()} gần {destination} ")

                if places_type == 'Khách sạn 🏨': 
                    df_hotel = df_place_rename[df_place_rename["Type"]=="Khách sạn"]
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
                            print(content)
                            iframe = folium.IFrame(content, width=300, height=125)
                            popup = folium.Popup(iframe, max_width=300)

                            icon_color = type_colour[row['Type']]
                            icon_type = type_icon[row['Type']]
                            icon = folium.Icon(color=icon_color, icon=icon_type)

                            # Use different icons for hotels, restaurants, and tourist attractions
                            folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)

                            st.write(f"## {index + 1}. {row['Name']}")
                            folium_static(mymap)
                            st.write(f"Đánh giá: {row['Rating']}")
                            st.write(f"Địa điểm: {row['Address']}")
                            st.write(f"Website: {row['Website_URL']}")
                            st.write(f"Các thông tin khác: {row['Google_Maps_URL']}\n")
                                
                elif places_type == 'Nhà hàng 🍴': 
                    df_restaurant = df_place_rename[df_place_rename["Type"]=="Nhà hàng"]
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
                            st.write(f"Đánh giá: {row['Rating']}")
                            st.write(f"Địa điểm: {row['Address']}")
                            st.write(f"Website: {row['Website_URL']}")
                            st.write(f"Các thông tin khác: {row['Google_Maps_URL']}\n")
                else:
                    df_tourist = df_place_rename[df_place_rename["Type"]=="Điểm du lịch"]
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
                            st.write(f"Đánh giá: {row['Rating']}")
                            st.write(f"Địa điểm: {row['Address']}")
                            st.write(f"Website: {row['Website_URL']}")
                            st.write(f"Các thông tin khác: {row['Google_Maps_URL']}\n")

            def chatbot():
                class Message(BaseModel):
                    actor: str
                    payload : str

                USER = "user"
                ASSISTANT = "ai"
                MESSAGES = "messages"

                # def initialize_session_state():
                if MESSAGES not in st.session_state:
                    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]
                
                msg: Message
                for msg in st.session_state[MESSAGES]:
                    st.chat_message(msg.actor).write(msg.payload)

                vector_dt = st.session_state["vector_dt"]
                
                qa = RetrievalQA.from_chain_type(
                    llm=TRAVEL_AGENT.travel_planning_agent.llm,
                    chain_type='stuff',
                    retriever=vector_dt[0].as_retriever(search_kwargs={"k": 8}),
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": vector_dt[1],
                        "memory": vector_dt[2]}
                )

                # Prompt
                query: str = st.chat_input("Enter a prompt here")

                if query:
                    st.session_state[MESSAGES].append(Message(actor=USER, payload=str(query)))
                    st.chat_message(USER).write(query)

                    with st.spinner("Please wait..."):
                        response: str = qa.invoke({"query": query})
                        response = response["result"]
                        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
                        # response = TRAVEL_AGENT.clean_response(response)
                        # for i, (name, infor) in enumerate(response.items()):
                        #     clean_response = f'{i+1}. {name}\n\nRating: {infor["Rating"]}\n\nReviewers: {infor["Number of user ratings"]}\n\nAddress: {infor["Address"]}\n\nPhone number: {infor["Phone number"]}\n\nWebsite: {infor["Website"]}'
                        list_place = response.split("\n\n")
                        if len(list_place) > 2:
                            mymap = folium.Map(location = st.session_state["initial_location"], 
                                        zoom_start=9, control_scale=True)
                        for place in list_place:
                            lat, lon = 0, 0
                            if "Vĩ độ" in place:
                                pattern = r'Vĩ độ: \d+\.\d+'
                                lat = regex.findall(pattern, place.strip(), regex.DOTALL)
                                pattern = r'\d+\.\d+'
                                lat = float(regex.findall(pattern, lat[0].strip(), regex.DOTALL)[0])
                            if "Kinh độ" in place:
                                pattern = r'Kinh độ: \d+\.\d+'
                                lon = regex.findall(pattern, place.strip(), regex.DOTALL)
                                pattern = r'\d+\.\d+'
                                lon = float(regex.findall(pattern, lon[0].strip(), regex.DOTALL)[0])
                                if "Nhà hàng" in place:
                                    type_place = "Nhà hàng"
                                if "Khách sạn" in place:
                                    type_place = "Khách sạn"
                                if "Điểm du lịch" in place:
                                    type_place = "Điểm du lịch"

                            if lat:
                                location = [lat, lon]
                                content = place.replace("-", "<br>")
                                print(content)
                                iframe = folium.IFrame(content, width=300, height=125)
                                popup = folium.Popup(iframe, max_width=300)

                                icon_color = st.session_state["type_colour"][type_place]
                                icon_type = st.session_state["type_icon"][type_place]
                                icon = folium.Icon(color=icon_color, icon=icon_type)

                                # Use different icons for hotels, restaurants, and tourist attractions
                                folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)
                        st.chat_message(ASSISTANT).write(response)
                        # Save map to HTML
                        map_html = mymap._repr_html_()  # not need save() to file
                        # Hiển thị trong khung chat
                        with st.chat_message(ASSISTANT):
                            st.markdown("📍 Bạn có thể xem bản đồ tham khảo dưới đây:")
                            html(map_html, height=500, width=700)
                   # st.write("Chatbot")

            method = st.sidebar.radio(" ",["Tìm kiếm 🔎","ChatBot 🤖","Dữ liệu 📑"], key="method_app")
            if method == "Tìm kiếm 🔎":
                maps()
            elif method == "ChatBot 🤖":
                chatbot()
            else:
                database()

if __name__ == '__main__':
    main()
    
# streamlit run app.py --server.port 8502 --server.address 0.0.0.0
