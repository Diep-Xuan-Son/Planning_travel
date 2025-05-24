import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
DIR = FILE.parents[0]
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

import os
import re
import jwt
import json
# import redis
import asyncio
import requests
import traceback
import re as regex
import pandas as pd
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Union

# from prompts import *
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompts import TRAVELING_AGENT_PROMPT, CITY_PROMPT

if not os.path.exists("./weights/nomic-embed-text-v1.5"):
    os.makedirs("./weights", exist_ok=True)
    print("----Downloading model!")
    # Name of the model on Hugging Face
    model_name = "nomic-ai/nomic-embed-text-v1.5"

    # Download and save to a custom folder
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save("./weights/nomic-embed-text-v1.5")  # Local folder

# Tool definitions
@dataclass
class ToolParameter:
    name: str
    description: str
    required: bool = False
    type: str = "string"

@dataclass
class ToolParameterHeader:
    content_type: str = "application/json"
    authorization: str = ""

@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    method: str = "GET"
    url: str = ""
    parametersHeaders: ToolParameterHeader = field(default_factory=ToolParameterHeader)
    next_action: str = ""
    function: Callable = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to the format expected by ChatGroq API."""
        param_properties = {}
        required_params = []
        
        for param in self.parameters:
            param_properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                param_properties[param.name]["enum"] = param.enum
            
            if param.required:
                required_params.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": param_properties,
                    "required": required_params
                }
            }
        }

    def to_promt(self, ) -> str:
        param_desc = ""
        required_params = []
        for param in self.parameters:
            param_desc += PARAM_PROMT.format(param_name=param.name, param_description=param.description, param_type=param.type) + "\n"
            if param.required:
                required_params.append(param.name)

        return param_desc, required_params

class Agent:
    def __init__(self, name: str, description: str, llm: object, tools: List[Tool] = None):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.conversation_history = []

        self.tools_name = []
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        self.tools_name.append(tool.name)
        return

    def to_promt(self, ) -> str:
        tool_desc = ""
        for tool in self.tools:
            param_promt, tool_required_params = tool.to_promt()
            tool_desc += TOOL_PROMT.format(tool_name=tool.name, tool_description=tool.description, tool_required_params=tool_required_params, param_promt=param_promt)
        return tool_desc

    def save_history(self, message):
        self.conversation_history.append(message)
        return

    def process_query(self, OutputStructured: object=None,  **kwargs) -> str:
        def OutputStructuredBase(BaseModel):
            """Format the response as JSON with value is text and key is 'result'"""
        prompt = self.description
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=f"You are a {self.name}"),
                HumanMessage(content=prompt.format(**kwargs))
            ]
        )
        if OutputStructured is not None:
            structured_output = self.llm.with_structured_output(OutputStructured)
        else:
            structured_output = self.llm.with_structured_output(OutputStructuredBase)
        chain = chat_prompt | structured_output
        result = chain.invoke({})
        return result

    def process_query_stream(self, query: str, context: str, is_stream: bool=True) -> str:
        res = self.llm(prompt=USER_PROMPT.format(query=query) + context, is_stream=is_stream)
        message_res = ""
        for r in res:
            # message_res += r
            # yield message_res
            yield r
        return

    async def call_tool(self, message) -> Dict:
        pattern = r'{.*}'
        clean_answer = regex.findall(pattern, message.replace("```", "").strip(), regex.DOTALL)
        if isinstance(clean_answer, list):
            clean_answer = clean_answer[0]
        clean_answer = eval(clean_answer)
        # print(f"----clean_answer: {clean_answer}")

        tool_dt = dict(zip(self.tools_name, self.tools))
        tasks = []
        tools_name = []
        tools_next_act = []
        results = {}

        for tn, pl in clean_answer.items():
            tool = tool_dt[tn]
            if not tool:
                return {"error": f"Tool '{tn}' not found or has no function implemented"}
            if not tool.url.startswith("http"):
                param = pl
            else:
                param = {
                    "method": tool.method, 
                    "url": tool.url, 
                    "headers": vars(tool.parametersHeaders), 
                    "params": pl
                }
            tasks.append(tool.function(**param))
            tools_name.append(tn)
            if tool.next_action:
                tools_next_act.append(f"Next action: " + tool.next_action + ", should finish doing the next action before call the tool {tn} again.")
            else:
                tools_next_act.append(tool.next_action)
        results = await asyncio.gather(*tasks)
        # print(results)
        # results = [f"{r}. {na}" for r, na in zip(results, tools_next_act)]
        results_next_act = " ".join(tools_next_act)
        results = dict(zip(tools_name, results))
        return results, results_next_act

class MyEmbeddings:
    def __init__(self, model):
        # self.model = TextEmbedding(model_name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
       self.model = SentenceTransformer(model, trust_remote_code=True)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # return list(self.model.query_embed(texts))
        return list(self.model.encode(texts))
            
    def embed_query(self, query: str) -> list[float]:
        # return list(self.model.query_embed([query]))
        return list(self.model.encode([query]))

def check_folder_exist(*args, **kwargs):
    if len(args) != 0:
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    if len(kwargs) != 0:
        for path in kwargs.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

# Sample tool implementations
class TravelAgentTool():
    def __init__(self, api_key: str, api_key_searchapi: str, dbmem_name: str, redis_url: str, redis_port: int):
        # init LLM
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key,  # if you prefer to pass api key in directly instaed of using env vars
                # base_url="...",
                # organization="...",
                # other params...,
                streaming=True
            )

        except:
            tb_str = traceback.format_exc()
            print(f"Error setup LLM: {tb_str}")
            raise "Internal error"

        self.api_key_geoapify = "ec8294c519294767ba0c6861434c3d88"
        self.api_key_searchapi = api_key_searchapi

        # self.dbmem_name = dbmem_name
        # self.num_mem = 20

        # self.redisClient = redis.StrictRedis(host=redis_url,
        #                       port=int(redis_port),
        #                       password="RedisAuth",
        #                       db=0)

        self.travel_planning_agent = Agent(
            name="Traveling Planning Assistant",
            description=TRAVELING_AGENT_PROMPT,
            llm = llm
        )

        self.city_agent = Agent(
          name="City Selection Assistant",
          description=CITY_PROMPT,
          llm = llm
        )

        # self.get_memory_agent = Agent(
        #   name="Get Memorry Assisstant",
        #   description=GET_MEMORY_AGENT,
        #   llm=llm
        # )

        with open("tools_data.json", "r", encoding="utf-8") as file:
            self.tool_data = json.load(file)
        # self.tools = {}
        self._prepare_tool()

    def _prepare_tool(self, ):
        for tool, attr in self.tool_data.items():
            self.travel_planning_agent.add_tool(Tool(
                    name=attr["name"],
                    description=attr["description"],
                    parameters=[ToolParameter(name=n, description=v["description"], required=v["required"], type=v["type"]) for n, v in attr["parameters"].items()] if attr["parameters"] is not None else [],
                    method=attr["method"],
                    url=attr["url"],
                    parametersHeaders=ToolParameterHeader(**attr["parametersHeaders"]),
                    next_action= attr["next_action"],
                    function=self.call_api
                ))

    @staticmethod
    async def call_api(method, url, headers, params):
        try:
            # res = requests.request(method, url=url, headers=headers, data=json.dumps(payload))
            res = requests.request(method, url=url, headers=headers, params=params)
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
            return(f"Tool cannot return the answer because Unable to establish connection with tool {url}.")

        if res.status_code == 422:
            return (f"Tool cannot return the answer because of missing prameter for tool")

        res = res.json()
        # print(f"----res: {res}")
        return res

    @staticmethod
    def clean_text(x):
        if isinstance(x, str):
            return x.replace('\xa0', ' ')
        elif isinstance(x, list):
            return [item.replace('\xa0', ' ') if isinstance(item, str) else item for item in x]
        return x

    @staticmethod
    def clean_response(response):
        pattern = r'{.*}'
        clean_answer = regex.findall(pattern, response["result"].replace("```", "").strip(), regex.DOTALL)
        
        if isinstance(clean_answer, list):
            clean_answer = clean_answer[0]
        clean_answer = eval(clean_answer)
        print(clean_answer)

        return clean_answer

    def prepare_data(self, destination: str="Thành phố Lạng Sơn", min_rating: float=4.0, radius: int=3000):
        destination_refined = self.city_agent.process_query(query=destination)
        destination = destination_refined["result"].lower()
        folder_data_test = f"{DIR}/data_test"
        folder_data_place = os.path.join(folder_data_test, destination.replace(" ", "_").replace(",", "_"))
        check_folder_exist(folder_data_place)

        if not os.path.exists(os.path.join(folder_data_place, f"destination.json")):
            print("----search new destination----")
            message_get_infor = {
                "Get infomation": {
                    "text": destination,
                    "country": "Việt Nam",
                    "limit": 1,
                    "lang": "vi",
                    "apiKey": self.api_key_geoapify
                }
            }
            message_get_infor = json.dumps(message_get_infor)
            tool_res, next_act = asyncio.run(self.travel_planning_agent.call_tool(message_get_infor))
            result_destination = tool_res["Get infomation"]
            with open(os.path.join(folder_data_place, f"destination.json"), "w") as f:
                json.dump(result_destination, f, indent=4)
        else:
            print("----get destination exist----")
            with open(os.path.join(folder_data_place, f"destination.json")) as f:
                result_destination = json.load(f)

        dt_destination = result_destination["features"][0]["properties"]
        print(dt_destination)
        # Create the circle
        circle_center = {"latitude": dt_destination["lat"], "longitude": dt_destination["lon"]}
        
        if not os.path.exists(os.path.join(folder_data_place, f"hotel_{radius}.json")):
            print("----search new data----")
            
            message_search = {
              "Search hotel": {
                  "engine": "google_maps",
                  "q": "Hotels",
                  "ll": f"@{circle_center['latitude']},{circle_center['longitude']},{radius}m",
                  "hl": "vi",
                  "api_key": self.api_key_searchapi
              },
              "Search restaurant": {
                  "engine": "google_maps",
                  "q": "Restaurants",
                  "ll": f"@{circle_center['latitude']},{circle_center['longitude']},{radius}m",
                  "hl": "vi",
                  "api_key": self.api_key_searchapi
              },
              "Search tourist": {
                  "engine": "google_maps",
                  "q": "Tourist Attractions",
                  "ll": f"@{circle_center['latitude']},{circle_center['longitude']},{radius}m",
                  "hl": "vi",
                  "api_key": self.api_key_searchapi
              },
            }
            message_search = json.dumps(message_search)
            tool_res, next_act = asyncio.run(self.travel_planning_agent.call_tool(message_search))

            result_hotel = tool_res["Search hotel"]
            with open(os.path.join(folder_data_place, f"hotel_{radius}.json"), "w") as f:
                json.dump(result_hotel, f, indent=4)

            result_restaurant = tool_res["Search restaurant"]
            with open(os.path.join(folder_data_place, f"restaurant_{radius}.json"), "w") as f:
                json.dump(result_restaurant, f, indent=4)

            result_tourist = tool_res["Search tourist"]
            with open(os.path.join(folder_data_place, f"tourist_{radius}.json"), "w") as f:
                json.dump(result_tourist, f, indent=4)
        else:
            print("----get data exist----")
            with open(os.path.join(folder_data_place, f"hotel_{radius}.json")) as f:
                result_hotel = json.load(f)
            with open(os.path.join(folder_data_place, f"restaurant_{radius}.json")) as f:
                result_restaurant = json.load(f)
            with open(os.path.join(folder_data_place, f"tourist_{radius}.json")) as f:
                result_tourist = json.load(f)

        df_hotel = pd.json_normalize(result_hotel['local_results'])
        df_hotel["type"] = "Khách sạn"

        df_restaurant = pd.json_normalize(result_restaurant["local_results"])
        df_restaurant["type"] = "Nhà hàng"

        df_tourist = pd.json_normalize(result_tourist['local_results'])
        df_tourist["type"] = "Điểm du lịch"

        df_place = pd.concat([df_hotel, df_restaurant, df_tourist], ignore_index=True)
        df_place = df_place.sort_values(by=['reviews', 'rating'], ascending=[False, False]).reset_index(drop=True)
        df_place = df_place.map(self.clean_text)
        df_place['combined_info'] = df_place.apply(lambda row: f"Type: {row['type']}, Name: {row['title']}. Rating: {row['rating']}. Number of user ratings: {row['reviews']}. Address: {row['address']}. Phone number: {row['phone']}. Latitude: {row['gps_coordinates.latitude']}. Longitude: {row['gps_coordinates.longitude']}. Website: {row['website']}", axis=1)
        df_place_rename = df_place.rename(columns={
            'type': 'Type',
            'title': 'Name',
            'address': 'Address',
            'phone': 'Phone',
            'rating': 'Rating',
            'reviews': 'User_Rating_Count',
            'reviews_link': 'Google_Maps_URL',
            'website': 'Website_URL',
            'gps_coordinates.latitude': 'Latitude',
            'gps_coordinates.longitude': 'Longitude',
            'thumbnail': 'Thumbnail'
        })
        df_place_brief = df_place_rename[['Type', 'Name', 'Address', 'Phone', 'Rating', 'User_Rating_Count','Google_Maps_URL', 'Website_URL', 'Latitude', 'Longitude', 'Thumbnail']]
        # print(df_place_brief)
        # print(df_place_brief[df_place_brief["Rating"] > 4])

        df_place_rename = df_place_rename[df_place_rename["Rating"] > 4]
        df_place_brief = df_place_brief[df_place_brief["Rating"] > 4]
        
        return df_place_rename, df_place_brief, circle_center

    def prepare_vector_data(self, data):
        # Load Processed Dataset
        loader = DataFrameLoader(data, page_content_column="combined_info")
        docs  = loader.load()

        # Document splitting
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="./weights/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', "trust_remote_code": True},        # hoặc 'cuda' nếu có GPU
            encode_kwargs={'normalize_embeddings': False}
        )
        # Vector DB
        vectorstore  = FAISS.from_documents(texts, embeddings)

        prompt = PromptTemplate(
            input_variables=["context","history","question"],
            template=self.travel_planning_agent.description,
        )

        memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

        # qa = RetrievalQA.from_chain_type(
        #     llm=self.travel_planning_agent.llm,
        #     chain_type='stuff',
        #     retriever=vectorstore.as_retriever(),
        #     verbose=True,
        #     chain_type_kwargs={
        #         "verbose": True,
        #         "prompt": prompt,
        #         "memory": memory}
        # )
        return vectorstore, prompt, memory

if __name__=="__main__":
    secret = "MMV"
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcGlfa2V5Ijoic2stcHJvai1QSDNHNnlMVEticmdvaU9ieTA4YlVMNHc0eVYxR3NJa25IeEltTl9VMFI1WmVsOWpKcDI0MzZuNUEwOTdVdTVDeXVFMDJha1RqNVQzQmxia0ZKX3dJTUw2RHVrZzh4eWtsUXdsMTN0b2JfcGVkV1c0T1hsNzhQWGVIcDhOLW1DNjY1ZE1CdUlLMFVlWEt1bzRRUnk2Ylk1dDNYSUEifQ.2qjUENU0rafI6syRlTfnKIsm6O4zuhHRqahUcculn8E'
    api_key_openai = jwt.decode(token, secret, algorithms=["HS256"])["api_key"]
    api_key_searchapi = "dZ3hmBQcvs1L4yDVbCjL8aXt"

    travel_agent = TravelAgentTool(api_key=api_key_openai, api_key_searchapi=api_key_searchapi, dbmem_name="Memory", redis_url="", redis_port=6400)
    df_place = travel_agent.prepare_data()

    vector_dt = travel_agent.prepare_vector_data(data=df_place[0])
    qa = RetrievalQA.from_chain_type(
        llm=travel_agent.travel_planning_agent.llm,
        chain_type='stuff',
        retriever=vector_dt[0].as_retriever(search_kwargs={"k": 8}),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": vector_dt[1],
            "memory": vector_dt[2]}
    )

    response: str = qa.invoke({"query": "đề xuất cho tôi một số nhà hàng"})
    print(response)
    print(response['result'].split("\n\n"))
    list_place = response['result'].split("\n\n")
    
    # list_place = ['Dưới đây là một số nhà hàng mà bạn có thể tham khảo:', '1. **Nhà hàng Áp Chao Xuân Sửu**\n   - Địa chỉ: 8 Phố Thân Thừa Quý, Vĩnh Trại, Thành phố Lạng Sơn, Lạng Sơn 240000, Việt Nam\n   - Số điện thoại: 0986 301 488\n   - Đánh giá: 4.2 (231 đánh giá)\n   - Vĩ độ: 21.8536395\n   - Kinh độ: 106.7593418\n   - Website: nan', '2. **Gà Đồi 207 Lê Lợi**\n   - Địa chỉ: 207 Lê Lợi, Vĩnh Trại, Thành phố Lạng Sơn, Lạng Sơn, Việt Nam\n   - Số điện thoại: 0356 797 688\n   - Đánh giá: 4.3 (44 đánh giá)\n   - Vĩ độ: 21.8547852\n   - Kinh độ: 106.7668261\n   - Website: nan', '3. **Ẩm thực Hồng Kông**\n   - Địa chỉ: 72 Lê Lợi, Vĩnh Trại, Thành phố Lạng Sơn, Lạng Sơn, Việt Nam\n   - Số điện thoại: 0966 583 686\n   - Đánh giá: 4.8 (12 đánh giá)\n   - Vĩ độ: 21.8543218\n   - Kinh độ: 106.7663923\n   - Website: nan', 'Hy vọng bạn tìm được nhà hàng ưng ý!']
    for place in list_place:
        if "Vĩ độ" in place:
            pattern = r'Vĩ độ: \d+\.\d+'
            lat = regex.findall(pattern, place.strip(), regex.DOTALL)
            pattern = r'\d+\.\d+'
            lat = regex.findall(pattern, lat[0].strip(), regex.DOTALL)
            print(lat)
        if "Kinh độ" in place:
            pattern = r'Kinh độ: \d+\.\d+'
            lon = regex.findall(pattern, place.strip(), regex.DOTALL)
            pattern = r'\d+\.\d+'
            lon = regex.findall(pattern, lon[0].strip(), regex.DOTALL)
            print(lon)
            if "Nhà hàng" in place:
                print("Nhà hàng")
            if "Khách sạn" in place:
                print("Khách sạn")
            if "Điểm du lịch" in place:
                print("Điểm du lịch")