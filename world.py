from agent import SimpleAgent, SimpleMemory
from typing import List, Any
import yaml 

# Handcoded (with occasional LLM assistance)

class Location:
  """
  Core Attributes:
    - name (str)
    - people (list of SimpleAgent objects present)
    - adjacent_locations (list of Locations)
    - attribute_dict (arbitrary dictionary, use for hackability)

  Core Methods
    - init
    - __str__ (provides default descriptions to agents and simulations)
    - description_dictionary() and description()
    - helper methods (set/unset adjacency, check for/place/remove agents)
  """
  name : str
  people : List[SimpleAgent]
  attribute_dict : dict | None

  def __init__(self, name, **kwargs) -> None:
    self.people = kwargs["agents"] if "agents" in kwargs else []
    self.adjacent_locations = kwargs["adjacent_locations"] if "adjacent_locations" in kwargs else []
    self.name = name

    self.attribute_dict = None
    
    # # Include pointer back to world.  Useful if, for example, you want to add items to both individual locations AND to a global list of items
    # self.world = kwargs["world"] if "world" in kwargs else []

  def __str__(self):
     return self.name

  def add_adjacent_location(self, location):
    self.adjacent_locations.append(location)

  def set_non_adjacent(self, location):
    if location in self.add_adjacent_location:
      self.adjacent_locations.pop(location)
    else:
       print(f"Warning: {self.name} and {location.name} are not adjacent, so connection cannot be removed.")

  def description(self):
    return yaml.dump(self.description_dictionary())

  def description_dictionary(self):
    adjacent_descriptions = []
    for loc in self.adjacent_locations:
      adjacent_descriptions.append(loc.name)

    person_descriptions=[]
    for person in self.people:
      # person_descriptions.append( person.description_dictionary() )
      person_descriptions.append(person.name)

    description : dict[str, Any] = {
      #  self.name : {
        "Name" : self.name,
        "Adjacent Locations" : adjacent_descriptions,
        "People Present" : person_descriptions,
      #  }
    }
    if self.attribute_dict is not None:
      for key,val in self.attribute_dict:
        description[key] = str(val)
    return description

  def contains_person(self, person):
    return person in self.people

  def add_person(self, person):
    if not(self.contains_person(person)):
      self.people.append(person)

  def remove_person(self, person):
    if self.contains_person(person):
      self.people.remove(person)

class World:
  """
  Core Attributes:
    - locations = list of locations
    - agents = list of all agents
  Core Methods
    - init
    - print/display world
    - adjacency operations (set/unset)
    - Functions to support Agent Tools
      - Access agent/location objects based on string
      - Backend functions to perform desired action (move agent, describe room)
        TODO: Move as much of the action functions as possible into the simulation?
  """
  locations : List[Location]
  agents: List[SimpleAgent]
  # things: List[Things]

  def __init__(self, **kwargs):
    """
    Required Args:
      location_names = list of strings 
      room_edges = list of pairs of location_names (possibly [])
      agent_descriptions = list of dictionaries with keys "name", "status", and "traits" attributes
    Optional Args: 
      model
    """

    self.model = kwargs["model"] if "model" in kwargs else None
    
    # Eventually: maybe need to initialize from existing locations/agents, but not for now!
    if "location_names" in kwargs and "room_edges" in kwargs and "agent_descriptions" in kwargs:
      location_names = kwargs["location_names"]
      list_of_adjacencies = kwargs["room_edges"]
      agent_description_list = kwargs["agent_descriptions"] 
      self.agents = []
      self.locations = []

      # Create room objects
      for loc_name in location_names:
        new_location = Location(loc_name, world = self )
        self.locations.append(new_location)

      # Set adjacencies      
      for adj in list_of_adjacencies:
        loc1_name = adj[0]
        loc2_name = adj[1]
        for loc1 in self.locations:
          if loc1.name == loc1_name:
            location1 = loc1
            break
        for loc2 in self.locations:
          if loc2.name == loc2_name:
            location2 = loc2
            break
        # Currently: undirected edges
        location1.add_adjacent_location(location2)
        location2.add_adjacent_location(location1)

      # Populate rooms with *NEW* agents
      for agent_dict in agent_description_list:
        name = agent_dict["name"]
        # persona = agent_dict["persona"]
        # status = agent_dict["status"]

        excluded_keys = ["name"]
        temp_agent_dict = {key:val for key,val in agent_dict.items() if key not in excluded_keys}

        if "memory" not in agent_dict: 
          temp_agent_dict["memory"] = SimpleMemory()
        if "model" not in agent_dict:
          temp_agent_dict["model"] = self.model

        if "location" in agent_dict:
          loc_name = agent_dict["location"]
          # print("agent in location", loc_name)
          for location in self.locations:
            if location.name == loc_name:
              temp_agent_dict["location"] = location
              # print("adding agent to location", loc_name)
              agent = SimpleAgent(name, **temp_agent_dict)
              self.agents.append(agent)
              location.add_person(agent)
              break
        else:
          print(f"Agent {name} has no location")
          agent = SimpleAgent(name, **temp_agent_dict)
          self.agents.append(agent)
    else: ## 
       print("Error: invalid location arguments to World()")
       print( """
              Required Args:
                location_names = list of strings 
                room_edges = list of pairs of location_names (possibly [])
                agent_descriptions = list of dictionaries with keys "name", "status", and "traits" attributes
              """)


  def print(self):
    location_descriptions = []
    for location in self.locations:
      loc_desc = location.description_dictionary()
      location_descriptions.append(loc_desc)
    print(yaml.dump(location_descriptions))


  

  def set_adjacent(self,location1 : Location, location2 : Location):
    if (location1 not in self.locations) | (location2 not in self.locations):
        print(f"Error: {location1.name} or {location2.name} not in World")
        return
    location1.add_adjacent_location(location2)
    location2.add_adjacent_location(location1)
    return 

  def set_not_adjacent(self,location1 : Location, location2 : Location):
    if (location1 not in self.locations) | (location2 not in self.locations):
        print(f"Error: {location1.name} or {location2.name} not in World")
        return
    location1.set_non_adjacent(location2)
    location2.set_non_adjacent(location1)
    return 

  # def add_agent(self, agent):
  #   self.agents.append(agent)
  # def add_location(self, location):
  #   self.locations.append(location)

  ## Helper functions for implementing Agent Tools
  ## (Convert string names to references to actual objects)
  def get_agent(self, agent_name : str) -> SimpleAgent | None:
      for agent in self.agents:
        if agent.name == agent_name:
           return agent
      print("Error: agent not found")
      return None
  
  def get_location(self, location_name : str) -> Location | None:
      for location in self.locations:
         if location.name == location_name:
            return location
      print(f"Location {location_name} not found")
      return 

  ## WORLD functions needed to implement AGENT tools
  ## NOTE:  some parameters (Agent object, Location object) not set by LLM
  ##        see simulation_test.py for tool definition and registration. 
  def move(self, **kwargs) -> str:
      """Args: 
          dest_name:str (set by LLM) 
          agent: SimpleAgent (set by Simulation)
      """
      dest_name = kwargs["dest_name"]  
      dest = self.get_location(dest_name)
      if dest is None:
         return f"Warning: no location of name {dest_name} found."

      agent = kwargs["agent"]         
      start = agent.get_location()
      if dest not in start.adjacent_locations:
          print(f"Warning: {dest_name} is not reachable from {start.name}.")
          return f"Failed: {agent.name} cannot reach {dest.name} from {start.name}"

      start.remove_person(agent)
      dest.add_person(agent)
      agent.set_location( dest )
      return f"Success: {agent.name} moved from {start.name} to {dest.name}."

  def describe_room(self, agent: SimpleAgent) -> str:
      location = agent.get_location()
      if location is None:
        return("You are NOT in any location.")
      names = ', '.join(p.name for p in location if p.name != agent.name) or "no-one"
      return f"You are in the {location}. People here: {names}."


   