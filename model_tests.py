
# Install google AI - pip install google-genai

from google.genai import types
from model import GeminiModel , Prompt

import pprint 

freemodel = GeminiModel("gemini-2.0-flash-lite", 30, 1000)

"""## Model Wrapper: Example Usage"""

# To define a tool

# # Step 1: write the function
# def get_biography(first_name:str, last_name:str):
#   return f"The biography of {first_name} {last_name} is ..."
# # Step 2: create the tool object
# tool = freemodel.register_tool(
#     get_biography,
#     "retrieves biography of specified individual",
#     {
#         "first_name": {"type" : "string", "description" : "The first or given name of an individual"},
#         "last_name" : {"type" : "string", "description" : "The last or family name of an individual"}
#     })
# print(tool)
# 
# 
## IMPORTANT: Run *MULTIPLE* tests to ensure CONSISTENT outcomes from the prompt choice
# num_tests = 10
# for i in range(num_tests):
#   print(f"Performing test {i+1} of {num_tests}")
#   # response = freemodel.generate_content( user_prompt="Write a short sentence about John Newton.", tools=[tool])
#   # freemodel.print_response(response)
#   # response = freemodel.generate_content( user_prompt="Write a short sentence about the weather.", tools=[tool])
#   # freemodel.print_response(response)
#   # response = freemodel.generate_content( user_prompt="Write a short sentence about the John Newton.", tools=[tool])
#   # freemodel.print_response(response)
#   response = freemodel.generate_content( user_prompt="Write a short sentence about first name John, second name Newton.  Use function calling as needed.", tools=[tool])
#   freemodel.print_response(response)
#   print('tool call result:' , freemodel.apply_tool(response))
#   # response = freemodel.generate_content( user_prompt="You are an expert agent, proficient both in natural language processing and in function calling.  Write a short sentence about John  Newton.  Use any tools needed to fulfil this request.", tools=[tool])
#   # freemodel.print_response(response)
#   print("--------------------------")

# Multimodal Tests

# Image
# response=freemodel.multimodal_query( user_prompt = "What is the color of this flower?", attachment_names = ["https://hips.hearstapps.com/hmg-prod/images/wisteria-in-bloom-royalty-free-image-1653423554.jpg"])
# print(response.text)
# # Video
# response=freemodel.multimodal_query( user_prompt = "What is the second joint in this video?", attachment_names =["https://www.youtube.com/watch?v=2bS_N6_QQos"])
# print(response.text)

# Prompt generation script test

# sample_prompt = Prompt(instruction = "say hello",
#                        persona = "angry,defensive",
#                        tone="friendly",
#                        output_format="JSON")
# prompt_text = sample_prompt.generate()
# print(prompt_text)
# response = freemodel.generate_content(user_prompt=prompt_text)
# pprint.pp(response.text)

sample_prompt = Prompt(persona = "You are an angry, defensive employee.",
                       tone="You are trying for a friendly tone.",
                       context="Your boss just said hello to you",
                       memory="You didn't sleep well last night.  You have been working hard on a project.  You think the project is going really well.",
                       instruction="What do you say?")
print("Prompt Object",sample_prompt)
import yaml
print("Yaml dump", yaml.dump(sample_prompt.details))
prompt_text = sample_prompt.generate()
print("Prompt Text", prompt_text)
response = freemodel.generate_content(user_prompt=prompt_text)
pprint.pp(response.text)


print("____________________")
from model import Prompt

spec = {
    "persona": "You are an adventurous marine biologist and gifted storyteller.",
    "context": "You are recording a segment for a children's science podcast about the ocean.",
    "instruction": "Explain why protecting coral reefs matters.",
    "input": [
        "Coral reefs support approximately 25% of all marine life",
        "They act as natural breakwaters that protect coastlines from storms",
        "Rising sea temperatures cause coral bleaching",
        "Pollution and overfishing accelerate reef decline"
    ],
    "tone": "inspiring and vivid",
    "output_format": "one paragraph of no more than 120 words"
}

prompt = Prompt(**spec)
prompt_text = prompt.generate()
print(prompt_text)      # inspect the YAML‑dump if curious

response = freemodel.generate_content(user_prompt=prompt_text)  # the model will call Prompt for you
print(response.text)