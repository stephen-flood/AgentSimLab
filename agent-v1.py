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



## Basic Prompt (o3's "distillation" of LangChain's GenerativeAgent experimental class)
RESPONSE_PROMPT_TEMPLATE = """
{summary}
It is {now}.
{agent_name}'s status: {status}

Location:
{location}

Summary of relevant memory:
{mem_summary}

Observation:
{observation}

What would {agent_name} do?

To end the conversation write:
GOODBYE: "…"
Otherwise continue the conversation with:
SAY: "…"
""".strip()


## Basic Memory Class (modified, simplified more)
class SimpleMemory:
  def __init__(self, memories: List[str] = None)  -> None:
    if memories is None:
      memories = []
    self._store: List[str] = memories

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

# Eventually, we will want the agent to be in one of many locations
# This will be defined later (so this code is usable on its own for now)
class Location:
  pass

## Basic Agent Class (almost exactly o3's output)
class SimpleAgent:
  def __init__(self,
               name:str ,
               **kwargs,
               ) -> None:
    """
    Optional arguments:
    - memory (SimpleMemory)
    - location (Location)
    - model (GeminiModel)
    - tools (list)
    - attribute_dictionary

    - persona (string)
    - status (string)
    """
    self.name = name
    special_attribute_keys = ["memory","model","tools","location"]
    self.memory = kwargs["memory"] if "memory" in kwargs else SimpleMemory()
    self.model = kwargs["model"] if "model" in kwargs else GeminiModel()
    self.tools = kwargs["tools"] if "tools" in kwargs else []
    self.location = kwargs["location"] if "location" in kwargs else None

    self.attribute_dictionary ={}

    if "attribute_dictionary" in kwargs:
      dict = kwargs["attribute_dictionary"]
      for key,val in dict.items():
        self.attribute_dictionary[key] = val

    # self.persona = kwargs["persona"] if "persona" in kwargs else None
    # self.traits  = kwargs["traits"] if "traits" in kwargs else None
    # self.status  = kwargs["status"] if "status" in kwargs else None
    
    # if "persona" in kwargs: self.persona = kwargs["persona"]
    # else: self.traits = ""
    # if "status" in kwargs: self.status = kwargs["status"]
    # else: self.status = ""
    # if "memory" in kwargs: self.memory = kwargs["memory"]
    # else: self.memory = SimpleMemory()
    # if "location" in kwargs: self.location = kwargs["location"]
    # else: self.location = None
    # if "model" in kwargs: self.model = kwargs["model"]
    # else: self.model = GeminiModel()
    # if "tools" in kwargs: self.tools = kwargs["tools"]
    # else: self.tools = []


  def generate_plan(self):
    pass
  def generate_action(self,**kwargs):
    if "instruction" not in kwargs:
      intruction = f"What would {self.name} do?"
    else:
      print("Error: duplicate instruction.")
    
    prompt_dictionary = {

    }

  def generate_speech(self, **kwargs):

    if "interlocutor_name" in kwargs:
      instruction = f"What would {self.name} say to {kwargs["interlocutor_name"]}?"
    else:
      instruction = f"What would {self.name} say?"
    # Include in every conversation
    prompt_dictionary = { 
      "name" : self.name ,
      "memories" : self.summarize_memory() ,
      "instruction" : instruction ,
      "location": self.location.name if self.location else "N/A" ,
      "description": self.description() ,
    }
    # Include addional information about the agent 
    # (everything stored in self.attribute_dictionary)
    for key,val in self.attribute_dictionary.items():
      if key not in prompt_dictionary or kwargs:
        prompt_dictionary[key] = val
    prompt = Prompt(**prompt_dictionary)
    prompt = prompt.generate()
    print(prompt)
    response = self.model.generate_content(user_prompt=prompt)
    print("Warning: need simulation to add conversation to *BOTH* parties.")    
    self.memory.add_memory(f"{self.name} reacted: {response.text}")
    return response.text

  def summarize_memory(self, num_memories=-1):
    mem_summary = self.memory.retrieve_recent_memories(num_memories)
    return mem_summary


  def _assemble_response_prompt(self, observation: str, num_memories: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # AUTOMATICALLY include memories
    # TODO:
    #   - Search with tool use
    #   - Differential memory (good/bad recall?  customer should know things the agent doesn't - starts with nonempty memory?)
    #
    return RESPONSE_PROMPT_TEMPLATE.format(
      summary=self.description,
      now=now,
      agent_name=self.name,
      status=self.status,
      mem_summary=self.summarize_memory(),
      observation=observation,
      location = self.location.name if self.location else "N/A",
    )
  

  def description_dictionary(self):
    """Returns dictionary that can be printed with yaml.dump"""
    description = {
      "Agent Name": self.name,
      # "Traits": self.traits,
      # "Status": self.status
    }
    description = description | self.attribute_dictionary
    if self.location is not None:
      description["Location"] = self.location.name
    return description

  def description(self):
    return yaml.dump(self.description_dictionary())

  def set_location(self, location):
    self.location = location


  def respond(self, observation: str, tools = None) -> Tuple[bool, str]:
    num_memories = -1
    prompt = self._assemble_response_prompt(observation,-1)
    llm_out = self.model.generate_content(user_prompt=prompt, tools = tools)
    
    # TEXT ONLY:
    ## TODO: Modify for tool use
    llm_out = llm_out.text
    print(llm_out)

    # simple parse
    say_match  = re.search(r'SAY:\s*"(.*)"', llm_out)
    bye_match  = re.search(r'GOODBYE:\s*"(.*)"', llm_out)

    if bye_match:
        utterance = bye_match.group(1)
        self.memory.add_memory(f"{self.name} said goodbye: {utterance}")
        # return False, f'{self.name} said "{utterance}"'
        return False, [self.name, utterance + "\n(GOODBYE)"]

    if say_match:
        utterance = say_match.group(1)
        self.memory.add_memory(f"{self.name} said: {utterance}")
        # return True, f'{self.name} said "{utterance}"'
        return True, [self.name, utterance]

    # fallback
    self.memory.add_memory(f"{self.name} reacted: {llm_out}")
    utterance = llm_out
    return False, ["Nobody", llm_out]
  
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

