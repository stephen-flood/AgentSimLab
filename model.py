# import base64
import os
from typing import List , Callable
from google import genai
from google.genai import types
from datetime import datetime
import time
import requests
import yaml # Used to quickly format prompts

## Import API key from Colab into OS environment
## (Required for code to load key from OS environment later)

try:
  from google.colab import userdata
  os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
except:
  with open("gemini.api_key") as file:
    os.environ["GEMINI_API_KEY"] = file.read()

# For parsing attachment names and types
from urllib.parse import urlparse
import mimetypes


## Define class to manage LLM queries to avoid rate limits on the free tier
class RateLimitTracker:
  def __init__(self, per_min_limit):
    self.per_min_limit = per_min_limit
    self.request_history_minute = []
  def log_query(self):
    self.request_history_minute.append(datetime.now())
    # remove old queries
    for time in self.request_history_minute:
      elapsed = datetime.now() - time
      if elapsed.seconds > 60:
        self.request_history_minute.remove(time)
  def print(self):
    print(self.request_history_minute)
  def time_to_wait(self):
    average_wait = 60 / self.per_min_limit
    # Spend the first quarter of the budget immediately
    # Spend the second quarter of the budget a bit quickly
    # Spend spend the second half slowly enough to avoid rate limiting
    if len(self.request_history_minute) <   (self.per_min_limit // 4):
      return 0
    elif len(self.request_history_minute) < (self.per_min_limit // 2):
      return average_wait / 2
    else:
      return average_wait * 2

class GeminiModel:
  """
  Wrapper for langauge model queries

  - Throttled queries to avoid rate limits on free tier with `generate_content`
  - Streamlined multimodal queries with `multimodal_query`
  - Simplified tool definition with `create_tool` function
  - (Partial) Handles tool calls
    - Currently only prints output or tool declartion with `print_response`
    - *TODO:*
      - Have `create_tool` also require a *FUNCTION* as an input
      - Have the GeminiModel object to STORE the functions and names internally
      - Have a `apply_tool` function that runs the relevant stored function
        definition on the tool identified byt he model
  """
  def __init__(self, model_name="gemini-2.0-flash-lite", rate_limit_minutes=30, rate_limit_daily=1000):
    self.model_name = model_name
    self.rate_limit_minutes = rate_limit_minutes
    self.rate_limit_daily = rate_limit_daily

    self.rate_limit_tracker = RateLimitTracker(rate_limit_minutes)
    self.API_KEY = os.environ.get("GEMINI_API_KEY")
    self.client = genai.Client(
        api_key=self.API_KEY,
    )

    # Store all tools created for this model (no duplicate names)
    # name suggested by o3
    self.tool_registry : dict[str, dict] = {}


  def generate_content(self, **kwargs):
    # Build a `config` object
    if "tools" in kwargs:
      tools = kwargs["tools"]
    else:
      tools = None
    generate_content_config = types.GenerateContentConfig(
        tools=tools, # Required for tool use
        response_mime_type="text/plain",
    )
    # print("Using tools", tools)
    # Build `contents`
    if "user_prompt" in kwargs:
      contents = [
          types.Content(
              role="user",
              parts=[types.Part.from_text(text=kwargs["user_prompt"]),],
          ),
      ]
    elif "contents" in kwargs:
      contents = kwargs["contents"]
    else:
      print("Error: no query given")
    # Keep track of queries to prevent rate limiting
    self.rate_limit_tracker.log_query()
    wait_time = self.rate_limit_tracker.time_to_wait()
    if wait_time > 0:
      print(f"Waiting {wait_time} seconds...")
      time.sleep(wait_time)
    # Query model, return response
    response = self.client.models.generate_content(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
    )
    return response

  def get_Part_from_name(self, ref :str):
      """
      Obtain an object for a file or URL that can be included in a Gemini API call.
      Args:
          ref (str): The reference or URL to the file
      """
      is_url = urlparse(ref).scheme in ("http", "https")
      if is_url:
          # check if the url is youtube
          if "youtube.com" in ref:
              mime = "video/mp4"
              return types.Part.from_uri(file_uri = ref, mime_type = mime)
          else:
            mime, _ = mimetypes.guess_type(ref)
            # check if it is an image
            # if mime.startswith("image/"):
            print("downloading file", ref)
            content_bytes = requests.get(ref).content
            object = types.Part.from_bytes(data=content_bytes, mime_type=mime)
            return object
      else:
          mime, _ = mimetypes.guess_file_type(ref)
          return types.Part.from_bytes(ref, mime)

  def multimodal_query(self, **kwargs):
    """
    Runs Gemini-2-flash model on multimodal user prompt.

    Args:
      prompt (str) : the instructions for the model
      attachment_name (List[str]) : A list of filenames and/or URL's
        that refer to files that should be attached in the query
    """

    if "user_prompt" in kwargs:
      prompt = kwargs["user_prompt"]
    else:
      print("Error: no prompt given")
      return

    if "system_prompt" in kwargs:
      system_prompt = kwargs["system_prompt"]
      sys_content = types.Content(
          role="system",
          parts=[types.Part.from_text(system_prompt)]
      )


    if "attachment_names" in kwargs:
      attachment_names = kwargs["attachment_names"]
    else:
      attachment_names = []

    content=[]
    for attachment_name in attachment_names:
      attachment_part = self.get_Part_from_name( attachment_name )
      content.append(attachment_part)

    if "system_prompt" in kwargs:
      content.append(sys_content)
    content.append(prompt)

    response = self.generate_content(
        model=self.model_name,
        contents=content,
    )
    return response

  # name and implementation suggested by o3
  def register_tool(self,
                    func : Callable,
                    description : str,
                    parameters  : dict[str,dict]):
    """
    registers a tool object for use with Gemini

    Args:
      function = name of a function already defined with
                  **NOTE:** type the name here (no parentheses!)
      description = "retrieves biography of specified individual",
      parameters = {
          "first_name": {"type" : "string",
                          "description" : "The first or given name of an individual"},
          "last_name" : {"type" : "string",
                          "description" : "The last or family name of an individual"}
      }

    IMPORTANT:
      Gemini appears to need a full, sentence long description of the argument for the tool to work.
      Even if it seems obvious, this MUST be done.  E.g.
                    "first_name": {"type" : "string", "description" : "The first or given name of an individual"},
      rather than
                    "first_name": {"type" : "string", "description" : "first or given name"},

    Example Usage (o3):
    def get_biography(first_name:str, last_name:str)->str:
      return f"{first_name} {last_name} was born …"       # whatever real logic you want

    bio_tool = freemodel.register_tool(
        func         = get_biography,
        description  = "Returns a short biography for a given person.",
        param_schema = {
            "first_name": {"type":"string",
                          "description":"The first (given) name of an individual."},
            "last_name" : {"type":"string",
                          "description":"The last (family) name of an individual."}
        }
    )

    Input Schema (o3):
    - `func`      : the Python function you want the model to call
    - `description`: one-sentence, natural-language description
    - `param_schema`: JSON schema **properties** block
                      (your `create_tool` already builds the wrapper)

    """
    name = func.__name__
    if name in self.tool_registry:
      print(f"Warning: Tool {name} already registered. Using PREVIOUS definition.")
      return self.tool_registry[name]["tool_object"]
    else:
      # tool_object = self.create_tool(name, description, parameters)
      # Create a gemini tool object from the given description
      tool_object = types.Tool(
        function_declarations=[
          types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=genai.types.Schema(
              type = genai.types.Type.OBJECT,
              properties = parameters,
              required = list(parameters.keys()),
            )
          )
        ]
      )

      # given a TEXT STRING name
      self.tool_registry[name] = {
        "function" : func, # Actual function to be called
        "tool_object" : tool_object, # Tool object for LLM call
        "description" : description, # text description
        "parameters" : parameters,   # dictionary of text attributes
      }
      return tool_object

  def get_tool(self,name):
    if name in self.tool_registry:
      return self.tool_registry[name]["tool_object"]
    else:
      print(f"Tool {name} not registered")
      return None
  
  def apply_tool(self, response):
    call_results = []
    if response.function_calls==None:
      call_results.append(("No Tool Calls",""))
      return call_results

    for call in response.function_calls:
      name = call.name
      args = call.args
      if name in self.tool_registry:
        func = self.tool_registry[name]["function"]
        result = func(**args)
        call_results.append((result, call))
      else:
        print(f"Tool {name} not registered in {call}")
    return call_results
  

  def print_response(self, response):
    #test if there was a function call
    if response.function_calls==None:
      print(response.text)
    else:
      print(response.function_calls)


class Prompt:
  """
  Assembles and formats prompts with containing the following parts
  persona, context, instruction, input, tone, output_format, examples
  """
  def __init__(self, **kwargs):
    # self.details = []
    # self.details.append({"Instruction" : instruction })
    self.details = {}

    # Put standard prompt elements in a specific order
    prompt_elements = ["persona","context","instruction","input","tone","output_format","examples"]
    for element in prompt_elements:
      if element in kwargs:
        name = element.title()
        name = name.replace("_"," ")
        # self.details.append({name: kwargs[element]})
        self.details[name] = kwargs[element]

    # Add all other prompt elements to the end 
    for key,val in kwargs.items():
      if key not in prompt_elements:
        self.details[key] = val

  def generate(self):
    # return yaml.dump(self.details, sort_keys=False)
    prompt = yaml.dump(self.details, sort_keys=False)
    # Repeat the instruction at the end for emphasis
    if "instruction" in self.details:
      prompt += f"\nRemember: the instruction is {self.details['instruction']}"
    return prompt
