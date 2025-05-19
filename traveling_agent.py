import os
import sys
import json
import re
import requests
import traceback
import asyncio
import re as regex
import redis
from typing import List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from openai import OpenAI, AuthenticationError

# from prompts import *
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


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

	def process_query(self, query: str, context: str, is_stream: bool=False) -> str:
		res = self.llm(prompt=USER_PROMPT.format(query=query) + context, is_stream=is_stream)
		message_res = ""
		for r in res:
			message_res += r
		return message_res

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
		print(f"----clean_answer: {clean_answer}")

		tool_dt = dict(zip(self.tools_name, self.tools))
		tasks = []
		tools_name = []
		tools_next_act = []
		results = {}

		for tn, pl in clean_answer.items():
			tool = tool_dt[tn]
			if not tool:
				return {"error": f"Tool '{tn}' not found or has no function implemented"}
			if not tool.url.startswith("http:/") or not tool.url.startswith("https:/"):
				param = pl
			else:
				param = {
					"method": tool.method, 
					"url": tool.url, 
					"headers": vars(tool.parametersHeaders), 
					"payload": pl
				}
			tasks.append(tool.function(**param))
			tools_name.append(tn)
			if tool.next_action:
				tools_next_act.append(f"Next action: " + tool.next_action + ", should finish doing the next action before call the tool {tn} again.")
			else:
				tools_next_act.append(tool.next_action)
		results = await asyncio.gather(*tasks)
		# print(results)
		results = [f"{r}. {na}" for r, na in zip(results, tools_next_act)]
		results_next_act = " ".join(tools_next_act)
		results = dict(zip(tools_name, results))
		return results, results_next_act

# Sample tool implementations
class TravelAgentTool():
	def __init__(self, api_key: str, dbmem_name: str, redis_url: str, redis_port: int):
		# init LLM
		try:
			llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=API_KEY,  # if you prefer to pass api key in directly instaed of using env vars
                # base_url="...",
                # organization="...",
                # other params...,
                streaming=True
            )

		except:
			tb_str = traceback.format_exc()
			LOGGER_AGENT.error(f"Error setup LLM: {tb_str}")
			raise "Internal error"

		# self.dbmem_name = dbmem_name
		# self.num_mem = 20

		# self.redisClient = redis.StrictRedis(host=redis_url,
		# 						port=int(redis_port),
		# 						password="RedisAuth",
		# 						db=0)

		TRAVELING_AGENT_PROMPT = """ 
        Your job is to assist users in locating a location. 
        From the following context and chat history, assist customers in finding what they are looking for based on their input. 
        Provide three recommendations, along with the address, phone number, website.
        Sort recommendations based on rating and number of user ratings. 
        
        {context}

        chat history: {history}

        input: {question} 
        Your Response:
        """
		self.travel_planning_agent = Agent(
			name="Traveling Planning Assistant",
			description=TRAVELING_AGENT_PROMPT,
			llm = llm
		)

		# self.synthesis_agent = Agent(
		# 	name="Synthesis Assistant",
		# 	description=SYNTHESIS_AGENT,
		# 	llm = llm
		# )

		# self.get_memory_agent = Agent(
		# 	name="Get Memorry Assisstant",
		# 	description=GET_MEMORY_AGENT,
		# 	llm=llm
		# )

		with open("tools_data.json", "r", encoding="utf-8") as file:
			self.tool_data = json.load(file)
		# self.tools = {}
		self._prepare_data()

	def _prepare_data(self, ):
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

		# self.sim_agent.description = self.sim_agent.description.format(tool_promt=self.sim_agent.to_promt())

	@staticmethod
	async def call_api(method, url, headers, payload):
		try:
			res = requests.request(method, url=url, headers=headers, data=json.dumps(payload))
		except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
			return(f"Tool cannot return the answer because Unable to establish connection with tool {url}.")

		if res.status_code == 422:
			return (f"Tool cannot return the answer because of missing prameter for tool")

		res = res.json()
		print(f"----res: {res}")
		return res['places']

	def prepare_data(self, destination, min_rating, radius):
		message_get_infor = {
			"Get infomation": {
	            "textQuery": destination,
	            "maxResultCount": 1,
			}
		}
		message_get_infor = json.dumps(message_get_infor)
		tool_res, next_act = asyncio.run(self.travel_planning_agent.call_tool(message_get_infor))
		# Convert JSON data to DataFrame
        df = pd.json_normalize(tool_res['Get infomation'])
        
        # Get the latitude and longitude values
        initial_latitude = df['location.latitude'].iloc[0]
        initial_longitude = df['location.longitude'].iloc[0]

        # Create the circle
        circle_center = {"latitude": initial_latitude, "longitude": initial_longitude}
        circle_radius = radius

		message_search = {
			"Search hotel": {
				"textQuery": f'Place to stay near {destination}',
				'minRating': min_rating,
                'locationBias': {
                    "circle": {
                        "center": circle_center,
                        "radius": circle_radius
                    }
                }
			},
			"Search restaurant": {
				"textQuery": f'Place to eat near {destination}',
				'minRating': min_rating,
                'locationBias': {
                    "circle": {
                        "center": circle_center,
                        "radius": circle_radius
                    }
                }
			},
			"Search tourist": {
				"textQuery": f'Tourist attraction near {destination}',
				'minRating': min_rating,
                'locationBias': {
                    "circle": {
                        "center": circle_center,
                        "radius": circle_radius
                    }
                }
			}
		}
		message_search = json.dumps(message_search)
		tool_res, next_act = asyncio.run(self.travel_planning_agent.call_tool(message_search))
		df_restaurant = pd.json_normalize(result_restaurant)


class SIMWorkFlow():
	def __init__(self, api_key: str, dbmem_name: str, redis_url: str, redis_port: int):
		# init LLM
		try:
			llm = OpenAILLM(api_key=api_key)
			is_valid_key = llm.check_openai_api_key()
			if not is_valid_key:
				raise "OpenAI API key is invalid!\n Try again with other API key." 
		except:
			tb_str = traceback.format_exc()
			LOGGER_AGENT.error(f"Error setup LLM: {tb_str}")
			raise "Internal error"

		self.dbmem_name = dbmem_name
		self.num_mem = 20

		self.redisClient = redis.StrictRedis(host=redis_url,
								port=int(redis_port),
								password="RedisAuth",
								db=0)

		self.sim_agent = Agent(
			name="SIM Assistant",
			description=SIM_AGENT_PROMPT,
			llm = llm
		)

		self.synthesis_agent = Agent(
			name="Synthesis Assistant",
			description=SYNTHESIS_AGENT,
			llm = llm
		)

		self.get_memory_agent = Agent(
			name="Get Memorry Assisstant",
			description=GET_MEMORY_AGENT,
			llm=llm
		)

		with open("tools_data.json", "r", encoding="utf-8") as file:
			self.tool_data = json.load(file)
		# self.tools = {}
		self._prepare_data()

	def _prepare_data(self, ):
		for tool, attr in self.tool_data.items():
			self.sim_agent.add_tool(Tool(
					name=attr["name"],
					description=attr["description"],
					parameters=[ToolParameter(name=n, description=v["description"], required=v["required"], type=v["type"]) for n, v in attr["parameters"].items()] if attr["parameters"] is not None else [],
					method=attr["method"],
					url=attr["url"],
					parametersHeaders=ToolParameterHeader(**attr["parametersHeaders"]),
					next_action= attr["next_action"],
					function=self.call_api
				))

		self.sim_agent.description = self.sim_agent.description.format(tool_promt=self.sim_agent.to_promt())

	@staticmethod
	async def call_api(method, url, headers, payload):
		try:
			res = requests.request(method, url=url, headers=headers, data=json.dumps(payload))
		except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
			return(f"Tool cannot return the answer because Unable to establish connection with tool {url}.")

		if res.status_code == 422:
			return (f"Tool cannot return the answer because of missing prameter for tool")

		res = res.json()
		LOGGER_AGENT.info(f"----res: {res}")
		return res["information"]

	def run(self, query, task_id):
		old_memory = []
		if self.redisClient.hexists(self.dbmem_name, task_id):
			old_memory = json.loads(self.redisClient.hget(self.dbmem_name, task_id))

		print(f"----query: {query}")
		message = self.sim_agent.process_query(query, self.sim_agent.description + HISTORY_PROMPT.format(memory=old_memory))
		tool_res, next_act = asyncio.run(self.sim_agent.call_tool(message))
		print(tool_res)

		synthesis_agent_description = self.synthesis_agent.description.format(tool_information=tool_res)
		message = self.synthesis_agent.process_query_stream(query, synthesis_agent_description)

		message_full = ""
		for word in message:
			yield word
			message_full += word
		memory_agent_description = self.get_memory_agent.description.format(message=message_full)
		memory = self.get_memory_agent.process_query(message_full, memory_agent_description)
		memory += " " + next_act
		print(f"----memory: {memory}")
		# if self.redisClient.hexists(self.dbmem_name, task_id):
		# 	old_memory = json.loads(self.redisClient.hget(self.dbmem_name, task_id))
		# 	print(f"----old_memory: {old_memory}")
		if old_memory:
			old_memory.append(memory)
			if len(old_memory)>self.num_mem:
				old_memory.pop(0)
			self.redisClient.hset(self.dbmem_name, task_id, json.dumps(old_memory))
		else:
			self.redisClient.hset(self.dbmem_name, task_id, json.dumps([memory]))
		return 

	# def run_batch(self, queries):
	# 	results = []
	# 	print(f"----queries: {queries}")
	# 	with ThreadPoolExecutor(max_workers=len(queries)) as executor:
	# 	# b = self.run(queries[0])
	# 	# print(b)
	# 	# executor = ThreadPoolExecutor(max_workers=len(queries))
	# 		# a = executor.submit(self.run, queries[0])
	# 		# print(a)
	# 		future_runner = {executor.submit(self.run, query): i for i, query in enumerate(queries)}
	# 		print(future_runner)
	# 		for runner in as_completed(future_runner):
	# 			runner_id = future_runner[runner]
	# 			try:
	# 				results.append(runner.result())
	# 				# LOGGER_AGENT.info(f"Successfully fetched data from {task_id}")
	# 			except Exception as e:
	# 				tb_str = traceback.format_exc()
	# 				LOGGER_AGENT.error(f"Error fetching data in process_query: {tb_str}")
	# 	return results

	async def run_batch(self, queries):
		results = []
		print(f"----queries: {queries}")
		tasks = [self.run(query) for query in queries]
		results = await asyncio.gather(*tasks)
		return results

if __name__=="__main__":
	# # Create travel agent
	# travel_agent = Agent(
	#     name="TravelAssistant",
	#     description="A helpful assistant for travel-related queries"
	# )

	# travel_agent_tool = TravelAgentTool()
	# print(travel_agent_tool.weather_tool)
	# parameters = {"location": "london", "unit": "celsius"}
	# result = travel_agent_tool.weather_tool.function(**parameters)
	# print(result)


	import time
	import jwt
	secret = "my sim chatbot"
	token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcGlfa2V5Ijoic2stcHJvai1weFVoZmkyZDY4QUphQXhIM0lWaFJJM0hfNXltVGFFYkRtZlJzRnltbEZVN0RHVTlteXZuNnkwbkE2c0x0WXpYSXdid09iYm9HWFQzQmxia0ZKUEFidThqbG1rM3RmUERCT2hYRjFkcWpRNDJKOVh4WFdSc2hyTi1tRGtlUTdkRVBhaDBJY0ViSXFSQXlvYTlxRzVRdHdUN3NMc0EifQ.Ozel4UKEN7_359DkrgwGIuuPIL07z3oKnv6WoK71_P0'
	api_key = jwt.decode(token, secret, algorithms=["HS256"])["api_key"]
	simw = SIMWorkFlow(api_key=api_key, dbmem_name="Memory", redis_url="192.168.6.163", redis_port=6400)
	# text_save = simw.sim_agent.description
	# with open("test.txt", "w") as f:
	# 	f.write(text_save)
	results = simw.run("show me my information", "abcd")
	message_res = ""
	for r in results:
		message_res += r
	print(message_res)
	# st_time = time.time()
	# asyncio.run(simw.run_batch(["show me my information"]))
	# print(f"----Duration: {time.time()-st_time}")