
# Install google AI - pip install google-genai

from google.genai import types
from model import GeminiModel 

freemodel = GeminiModel("gemini-2.0-flash-lite", 30, 1000)

"""## Model Wrapper: Example Usage"""

# To define a tool

# Step 1: write the function
def get_biography(first_name:str, last_name:str):
  return f"The biography of {first_name} {last_name} is ..."
# Step 2: create the tool object
tool = freemodel.register_tool(
    get_biography,
    "retrieves biography of specified individual",
    {
        "first_name": {"type" : "string", "description" : "The first or given name of an individual"},
        "last_name" : {"type" : "string", "description" : "The last or family name of an individual"}
    })
# print(tool)


## IMPORTANT: Run *MULTIPLE* tests to ensure CONSISTENT outcomes from the prompt choice
num_tests = 10
for i in range(num_tests):
  print(f"Performing test {i+1} of {num_tests}")
  # response = freemodel.generate_content( user_prompt="Write a short sentence about John Newton.", tools=[tool])
  # freemodel.print_response(response)
  # response = freemodel.generate_content( user_prompt="Write a short sentence about the weather.", tools=[tool])
  # freemodel.print_response(response)
  # response = freemodel.generate_content( user_prompt="Write a short sentence about the John Newton.", tools=[tool])
  # freemodel.print_response(response)
  response = freemodel.generate_content( user_prompt="Write a short sentence about first name John, second name Newton.  Use function calling as needed.", tools=[tool])
  freemodel.print_response(response)
  print('tool call result:' , freemodel.apply_tool(response))
  # response = freemodel.generate_content( user_prompt="You are an expert agent, proficient both in natural language processing and in function calling.  Write a short sentence about John  Newton.  Use any tools needed to fulfil this request.", tools=[tool])
  # freemodel.print_response(response)
  print("--------------------------")

# Multimodal Tests

# Image
response=freemodel.multimodal_query( user_prompt = "What is the color of this flower?", attachment_names = ["https://hips.hearstapps.com/hmg-prod/images/wisteria-in-bloom-royalty-free-image-1653423554.jpg"])
print(response.text)
# Video
response=freemodel.multimodal_query( user_prompt = "What is the second joint in this video?", attachment_names =["https://www.youtube.com/watch?v=2bS_N6_QQos"])
print(response.text)
