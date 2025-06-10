
# Install google AI - pip install google-genai

from google.genai import types
from model import GeminiModel, HTTPChatModel, HFTransformersModel
from model import Prompt

import pprint 

# # Gemma models, google API: 
# # NO native tool use
# # NO system prompt :(
# freemodel = GeminiModel(
#     "gemma-3-27b-it",
#     # "gemma-3n-e4b-it",
#     30,    
#     native_tool = False,
#     verbose = False,
#     allow_system_prompt = False,
# )

# # Gemini models, google API ONLY
# # compare native/non-native tool use
# freemodel = GeminiModel(
#     "gemini-2.0-flash-lite", 
#     30,
#     native_tool = True,
#     verbose=False,
#     )

# # Any OpenAPI compatible provider
freemodel = HTTPChatModel(
# # Models:
  ## WITH native Tools
    # "mistral-small:24b-instruct-2501-q4_K_M",
    # native_tool=True, 
  ## NO native tools
    "gemma3:12b", 
    native_tool=False, 
  ## Other Flags
    multimodal=True, 
    verbose=False,
)

# freemodel = HFTransformersModel(
#   # "microsoft/Phi-4-mini-instruct",
#   "allenai/OLMo-2-0425-1B-Instruct",
#   native_tool=False,
# )

"""## Model Wrapper: Example Usage"""

test_prompt = True
test_tools = True
test_multimodal = True
test_history = True

if test_prompt :
  print("============== PROMPT TEST 1 ==============")

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
  pprint.pp(freemodel.response_text(response))



  print("============== PROMPT TEST 2 ==============")

  print("--------------------------")
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
  # print(response.text)
  pprint.pp(freemodel.response_text(response))


  # Prompt generation script test

  # sample_prompt = Prompt(instruction = "say hello",
  #                        persona = "angry,defensive",
  #                        tone="friendly",
  #                        output_format="JSON")
  # prompt_text = sample_prompt.generate()
  # print(prompt_text)
  # response = freemodel.generate_content(user_prompt=prompt_text)
  # pprint.pp(response.text)


if test_tools:

  print("============== TOOL TESTS ==============")
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
  print(tool)


  # IMPORTANT: Run *MULTIPLE* tests to ensure CONSISTENT outcomes from the prompt choice
  num_tests = 10
  for i in range(num_tests):
    print(f"Performing test {i+1} of {num_tests}")
    response = freemodel.generate_content( user_prompt="Write a short sentence about first name John, second name Newton.  Use function calling as needed.", tools=[tool])
    tool_call_list = freemodel._iter_tool_calls(response)
    if tool_call_list is not None:
      for call in tool_call_list:
        print(call)
    else: 
      print("WARNING: No tool calls")
      print(freemodel.print_response(response))
    # freemodel.print_response(response)
    # print('\nModel response:\n',freemodel.response_text(response))
    ## Observe *RESULT* of calling the tool functions
    # print('\nTool call result:' , freemodel.apply_tool(response))

    # print(response)
    print("--------------------------")

if test_multimodal:
  print("============== MULTIMODAL TESTS ==============")
  
  print("Test 1: local image file")
  from pathlib import Path
  import io, requests
  from PIL import Image

  img_path = Path("random_image.jpg")
  if not img_path.exists():
      # Downloaded from picsum.photos
      # the `laurem ipsum` of photos
      # single number after slash provides 
      #   square image of that size
      jpg = requests.get("https://picsum.photos/256").content
      Image.open(io.BytesIO(jpg)).save(img_path)
      print("Saving new image: `random_image.jpg`")
  response = freemodel.generate_content(
      user_prompt="Describe the image in one sentence.",
      attachment_names=[str(img_path)],
  )
  print(freemodel.response_text(response))

  # Hosted Image
  print("Test 2: remote image file")
  response=freemodel.generate_content( user_prompt = "What is the color of this flower?", attachment_names = ["https://hips.hearstapps.com/hmg-prod/images/wisteria-in-bloom-royalty-free-image-1653423554.jpg"])
  print(freemodel.response_text(response))

  # Online Video
  print("Test 3: remote video file")
  response=freemodel.generate_content( user_prompt = "What is the second joint in this video?", attachment_names =["https://www.youtube.com/watch?v=2bS_N6_QQos"])
  print(freemodel.response_text(response))


if test_history:
  print("============== HISTORY TESTS ==============")
  history = [
      {"role": "user",      "content": "Who won the 2022 World Cup?"},
      {"role": "assistant", "content": "Argentina lifted the trophy."},
  ]

  resp = freemodel.generate_content(
      user_prompt="And who scored the winning goal?",
      history=history
  )
  print(freemodel.response_text(resp))
