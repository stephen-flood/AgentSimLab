from typing import List, Tuple
import re # Parse model response to decide if we should end the conversation.
          # TODO: replace this with a tool call?
from datetime import datetime
from model import GeminiModel, Prompt, RateLimitTracker

## Use to export dictionaries/lists in nice readable format
#pip install pyyaml
import yaml 

## Formatting output
try :
  # pip install rich
  # from rich import print
  from rich import Console
  def show(speaker, text):
      from rich.console import Console
      console = Console()          # auto‑detects terminal width
      console.print(f"[bold cyan]{speaker}[/]: {text}")   # auto‑wraps & colours
      #
      # print(f"[bold]{speaker}[/]: {text}")   # auto‑wraps & colours
except:
  def show(speaker, text):
      print(f"*{speaker}*: {text}")  
  pass 


## Basic Memory Class (modified, simplified more)
class SimpleMemory:
  def __init__(self, memories: List[str] = None)  -> None:
    if memories is None:
      memories = []
    self._store: List[str] = memories

  ## Do this if a SimpleMemory object is converted to a string
  ##  or used in a string context
  def __str__(self):
    return yaml.dump(self.retrieve_recent_memories(-1),sort_keys=False)

  def add_memory(self, text: str) -> None:
    self._store.append(text)

  def retrieve_recent_memories(self, n) -> str:
    """
    Returns the most recent n memories.

    Returns all memories if n < 0
    """
    if n < 0:
      return self._store
    elif len(self._store) < n:
      return self._store
    else:
      return self._store[-n:]

class Location:
  pass

## Basic Agent Class (almost exactly o3's output)
class SimpleAgent:
  def __init__(self,
               name:str ,
               **kwargs,
               ) -> None:
    """
    Args: 
    - name (str): required, must be unique
    - list of all other attributes
      - required: model (GeminiModel), tools (List)
      - optional: any other attribute you'd like the model to store
                  (eg persona, memory, location, etc.)
      - IMPORTANT: each `optional` object MUST have a __str__ function defined for its object
    """
    self.name = name
  
    special_attribute_keys = ["model","tools"]
    # When subclassing, expand to other types of 'specially handled' variables
    # special_attribute_keys = ["model","tools","location","memory"]
    self.model = kwargs["model"] if "model" in kwargs else GeminiModel()
    self.tools = kwargs["tools"] if "tools" in kwargs else []

    if "attribute_dictionary" in kwargs:
     self.attribute_dictionary = kwargs["attribute_dictionary"]
    else:
      self.attribute_dictionary ={}

    for key,val in kwargs.items():
      if key not in special_attribute_keys:
        self.attribute_dictionary[key] = val

  def __str__(self):
    return self.description()

  def generate_content(self,**kwargs):
    """
    Generate content using a prompt made up of 
        1. Internal attributes, and 
        2. Passed arguments
    Prompt string will be genrated using yaml.dump

    Arguments: 
      - A dictionary of any/all fields you want to appear in the prompt
        (dictionary will be converted to YAML output)
    Notes
      - (optional) "model_parameters" (dict) = dictionary of model objects you want passed to the LLM call
      - (behavior) Override internal attributes by providing option with same key in kwargs

    IMPORTANT:
      - (all objects stored in attribute_dictionary MUST have a __str__ method).
    """
    special_args = ["model_parameters"]
    model_params = kwargs["model_parameters"] if "model_parameters" in kwargs else {}

    arg_dict = {key:str(val) for key,val in kwargs.items() if key not in special_args }
    attribute_dict = {key:str(val) for key,val in self.attribute_dictionary.items()}
    # Generate prompt from union of agent attributes and  kwargs
    #  Kwargs comes second, so overrides agent attributes if conflict
    prompt_dictionary = attribute_dict | arg_dict 
    prompt = Prompt(**prompt_dictionary)
    prompt = prompt.generate()
    # print(prompt)
    response = self.model.generate_content(user_prompt=prompt, **model_params)
    return response


  def generate_plan(self,**kwargs):
    pass

  def generate_action(self,**kwargs):
    """
    Sets *ACTION* instruction, then calls generate_content, passing all kwargs
    """
    prompt_dict = kwargs
    prompt_dict["name"] = self.name
    prompt_dict["description"] = self.description() 
    
    prompt_dict["instruction"] = f"What would {self.name} do? "\
                                  "State the action and describe its results."

    if "location" in self.attribute_dictionary:
      prompt_dict["location description"] = self.attribute_dictionary["location"].description()

    response = self.generate_content(**prompt_dict)    

    return response

  def generate_speech(self, **kwargs):
    """
    Sets *SPEECH* instruction, then calls generate_content, passing all kwargs
    """
    prompt_dict = kwargs
    prompt_dict["name"] = self.name
    prompt_dict["description"] = self.description() 

    if "interlocutor_name" in kwargs:
      prompt_dict["instruction"] = f"What would {self.name} say to {kwargs["interlocutor_name"]}?"
    else:
      prompt_dict["instruction"] = f"What would {self.name} say?"

    if "location" in self.attribute_dictionary:
      prompt_dict["location description"] = self.attribute_dictionary["location"].description()

    response = self.generate_content(**prompt_dict)    
    return response.text



  def description_dictionary(self):
    """Returns dictionary that can be printed with yaml.dump"""
    description = {
      "Agent Name": self.name,
    }
    description = description | self.attribute_dictionary
    # if "location" is not None:
    #   description["Location"] = self.location.name
    return description



  def description(self):
    return yaml.dump(self.description_dictionary())


  def set_location(self, location):
    self.location = location


  ## Copied from "GoogleModel" definition
  ## Needed here for simulations, where the actions that can be taken
  ## will depend on the agent calling them 
  def apply_tool_calls(self, response):

    call_results = []
    if response.function_calls==None:
      call_results.append(("No Tool Calls",""))
      return call_results

    for call in response.function_calls:
      name = call.name
      args = call.args
      if "agent" in args:
        print("Error: Model cannot set acting agent")
        exit
      else:
        args["agent"] = self
      if name in self.model.tool_registry:
        func = self.model.tool_registry[name]["function"]
        result = func(**args)
        call_results.append((result, call))
      else:
        print(f"Tool {name} not registered in {call}")
    return call_results

