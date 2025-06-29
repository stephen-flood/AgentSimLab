from typing import List, Tuple
import re # Parse model response to decide if we should end the conversation.
          # TODO: replace this with a tool call?
from datetime import datetime
from model import SimpleModel, GeminiModel, Prompt, RateLimitTracker
import yaml 

## Basic Memory Class (modified, simplified more)
class SimpleMemory:
  def __init__(self, memories: List[str] | None = None)  -> None:
    if memories is None:
      memories = []
    self._store: List[str] = memories

  ## Do this if a SimpleMemory object is converted to a string
  ##  or used in a string context
  def __str__(self):
    if len(self._store) == 0 :
      return "N/A"
    return yaml.dump(self.retrieve_recent_memories(-1),sort_keys=False)

  def add_memory(self, text: str) -> None:
    self._store.append(text)

  def clear_memory(self):
    self._store = []

  def retrieve_recent_memories(self, n) -> List[str]:
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

## Variant of SimpleMemory that automatically compresses itself using an agent's LLM
class SelfCompressingMemory():
  def __init__(self, max_chars, model, memories: List[str] | None = None, **kwargs):
    self.max_chars = max_chars
    self.model = model
    self.verbose = False

    # Set persona and instruction
    default_persona = "You are an agent with limited memory.  You need to process the YAML representation of your current memory to REDUCE the number of characters below your maximum threshold."
    self.persona = kwargs["persona"] if "persona" in kwargs else default_persona
    #
    default_instruction = "Write down the most relevant information from your MEMORY in YAML format.  Use your MEMORY for detailed information about your task."
    self.instruction = kwargs["instruction"] if "instruction" in kwargs else default_instruction


    if memories is None:
      memories = []
    self.memories: List[str] = memories

    self.length = 0
    for memory in self.memories:
      self.length += len(memory)

  def add_memory(self, text:str ):
    self.memories.append(text)
    self.length += len(text)
    # check if memory exceeds maximum size
    memory_yaml = str(self)
    if len(memory_yaml) > self.max_chars:
      print(f"Memory size of {len(memory_yaml)} exceeds maximum of {self.max_chars} characters.  Compressing.")
      prompt = Prompt(
        persona = self.persona,
        max_chars = self.max_chars,
        instruction = self.instruction,
        memories = memory_yaml,
      ).generate()
      compressed_memory_response = self.model.generate_content(user_prompt=prompt)
      compressed_memory = self.model.response_text(compressed_memory_response)

      self.length = len(compressed_memory)

      try:
        # add_memory assumes existence of self.memories.append(...)
        # but an LLM might output a single string which parses as a string (not a list)
        self.memories = [yaml.safe_load(compressed_memory)]
      except:
        self.memories = [compressed_memory]

      if self.length > self.max_chars:
        print("ERROR: memory compression FAILED.  Produce a shorter summary next time.")
        if self.verbose: 
          print(compressed_memory) 
          raise ValueError("Failure compressing memory.")

      if self.verbose:
        print("Compressing memories")
        print(compressed_memory)


  def clear_memory(self):
    self.memories = []
    self.len = 0

  def __str__(self):
    if len(self.memories) == 0 :
      return "N/A"
    return yaml.dump(self.memories,sort_keys=False)


## Basic Agent Class (first draft by o3, then refactored)
class SimpleAgent:
  def __init__(self,
               name:str ,
               plan_instruction_template:   str | None = None,
               action_instruction_template: str | None = None,
               speech_instruction_template: str | None = None,
               **kwargs,
               ) -> None:
    """
    Args: 
    - name (str): required, must be unique

    - Special
      - plan_instruct_template = instruction template provided on .generate_plan(...)
      - action_instruct_template = instruction template provided on .generate_action(...)
      - speech_instruct_template = instruction template provided on .generate_speech(...)
      - tools = list of tool objects registered with the model

    - list of all other attributes
      - required: model (SimpleModel), 
      - optional: any other attribute you'd like the model to store
        - persona, memory, location, memory, etc

    - IMPORTANT: each `optional` object MUST have a __str__ function defined for its object

    """
    self.name = name
  
    special_attribute_keys = ["model","tools"]
    # When subclassing, expand to other types of 'specially handled' variables
    self.model = kwargs["model"] if "model" in kwargs else GeminiModel()
    self.tools = kwargs["tools"] if "tools" in kwargs else []
    # "location" and "memory" are not special because they have a __str__ method that is called in prompting and describing

    self.verbose = False

    if self.verbose: print(kwargs)

    default_plan_instruct_template = "First, identify what {self.name} would do.  Then make a very short plan to achieve those goals.  Find a SMALL NUMBER of concrete steps that can be taken.  Take available tools into account in your planning, but DO NOT do any tool calls."
    default_action_instruct_template = "What would {self.name} do? "
    default_speech_instruct_template = "What would {self.name} say?"
    self.plan_instruct_template   = plan_instruction_template   or default_plan_instruct_template
    self.action_instruct_template = action_instruction_template or default_action_instruct_template
    self.speech_instruct_template = speech_instruction_template or default_speech_instruct_template


    if "attribute_dictionary" in kwargs:
     self.attribute_dictionary = kwargs["attribute_dictionary"]
    else:
      self.attribute_dictionary ={}

    for key,val in kwargs.items():
      if key not in special_attribute_keys:
        self.attribute_dictionary[key] = val

  def __str__(self):
    return self.description()
  
  def add_memory(self, 
                 memory_string : str, 
                 memory_variable : str | None  = None):
    if memory_variable == None:
      memory_variable = "memory"
    if memory_variable in self.attribute_dictionary:
      self.attribute_dictionary[memory_variable].add_memory( memory_string )
      return 
    else:
      raise ValueError(f"Error: {self.name} has no memory object to store {memory_string}")

  def clear_memory(self, memory_variable : str | None = None):
    if memory_variable == None:
      memory_variable = "memory"
    
    if memory_variable in self.attribute_dictionary:
      self.attribute_dictionary[memory_variable].clear_memory()
      return 
    else:
      raise ValueError(f"Error: {self.name} has no memory object to clear")

  def generate_plan(self,**kwargs):    
    """
    Sets *PLAN* instruction, then calls generate_content, passing all kwargs
    """
    prompt_dict = kwargs
    prompt_dict["name"] = self.name
    prompt_dict["instruction"] = self.plan_instruct_template.format(self=self, kwargs=kwargs)
    response = self.generate_content(**prompt_dict)    
    return self.model.response_text(response)


  def generate_action(self,**kwargs):
    """
    Sets *ACTION* instruction, then calls generate_content, passing all kwargs
    """
    prompt_dict = kwargs
    prompt_dict["name"] = self.name
    prompt_dict["instruction"] = self.action_instruct_template.format(self=self, kwargs=kwargs)
    response = self.generate_content(**prompt_dict)    

    return response

  def generate_speech(self, **kwargs):
    """
    Sets *SPEECH* instruction, then calls generate_content, passing all kwargs
    """
    prompt_dict = kwargs
    prompt_dict["name"] = self.name
    prompt_dict["instruction"] = self.speech_instruct_template.format(self=self, kwargs=kwargs)
    # if "interlocutor_name" in kwargs:
    #   prompt_dict["instruction"] = f"What would {self.name} say to {kwargs["interlocutor_name"]}?"
    # else:
    #   prompt_dict["instruction"] = f"What would {self.name} say?"
    response = self.generate_content(**prompt_dict)    
    return self.model.response_text(response)

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
    special_args = ["model_parameters","tools"]
    if "model_parameters" in kwargs:
      model_params = kwargs["model_parameters"] 
    elif "tools" in kwargs:
      model_params = {"tools":kwargs["tools"]}
    else:
      model_params = {}

    arg_dict = {key:str(val) for key,val in kwargs.items() if key not in special_args }
    attribute_dict = {key:str(val) for key,val in self.attribute_dictionary.items()}
    # Generate prompt from union of agent attributes and  kwargs
    #  Kwargs comes second, so overrides agent attributes if conflict
    prompt_dictionary = attribute_dict | arg_dict 
    prompt = Prompt(**prompt_dictionary)
    prompt = prompt.generate()

    if self.verbose: print(prompt)

    response = self.model.generate_content(user_prompt=prompt, **model_params)
    return response

  def description_dictionary(self):
    """ Converts all entries in attribute_dictionary to STRINGS, plus adds agent name """
    description = {
      "Agent Name": self.name,
    }
    ## OPTIONAL: add FULL room information to agent description
    # if "location" is not None:
    #   description["Location Description"] = self.location.description()
    str_attribute_dict = {key:str(val) for key,val in self.attribute_dictionary.items()}
    description = description | str_attribute_dict
    return description

  def description(self):
    return yaml.dump(self.description_dictionary())

  ## Methods for SIMULATION to move agent around
  def get_location(self):
    return self.attribute_dictionary["location"] if "location" in self.attribute_dictionary else None

  def set_location(self, location):
    self.attribute_dictionary["location"] = location

  ## Methods for SIMULATION to update agent attributes
  ## e.g. changing task, adding/removing items, etc.
  def get_attribute(self, key):
    return self.attribute_dictionary[key] if key in self.attribute_dictionary else None
  
  def set_attribute(self, key, value):
    self.attribute_dictionary[key] = value
