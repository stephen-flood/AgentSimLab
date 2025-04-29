from typing import List

from agent import SimpleAgent, SimpleMemory

## Use to export dictionaries/lists in nice readable format
#pip install pyyaml
import yaml 

# Use to PAD exports
def indent(text: str, num_spaces: int) -> str:
    """
    Return a new string where each '\n' in `text` is preceded by `num_spaces` spaces.
    """
    spacer = ' ' * num_spaces
    # replace each newline with spaces + newline
    return text.replace('\n', spacer + '\n')


# Handcoded (with occasional LLM assistance)

class Location:
  name : str
  people : List[SimpleAgent]
  general_description : str
  # things : List[Thing]
  # adjacent_locations : List[Location]

  ## Create/modify location
  def __init__(self, name, people_present=None, adjacent_locations=None) -> None:
    if people_present is None:
      people_present = []
    if adjacent_locations is None:
      adjacent_locations = []
    self.general_description = ""
    self.name = name
    self.people = people_present
    self.adjacent_locations = adjacent_locations

  def add_adjacent_location(self, location):
    self.adjacent_locations.append(location)

  # def remove_adjacent_location(self, location):
  #   self.adjacent_locations.remove(location)

  def description(self):
    return yaml.dump(self.description_dictionary(), indent=2)

  def description_dictionary(self):
    adjacent_descriptions = []
    for loc in self.adjacent_locations:
      # adjacent_descriptions += f"{loc.name}:{loc.general_description},"
      adjacent_descriptions.append(loc.name)

    person_descriptions=[]
    for person in self.people:
      person_descriptions.append( person.description_dictionary())

    description = {
       self.name : {
        "Adjacent Locations" : adjacent_descriptions,
        "People Present" : person_descriptions,
       }
    }
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
  locations : List[Location]
  agents: List[SimpleAgent]
  # things: List[Things]

  def __init__(self, locations=None, agents=None , model = None) -> None:

    if locations is None:
      locations = []
    if agents is None:
      agents = []
    self.locations = locations
    self.agents = agents
    self.model = model

  def clear(self):
    self.locations = []
    self.agents = []

  def print(self):
    location_descriptions = []
    for location in self.locations:
      loc_desc = location.description_dictionary()
      location_descriptions.append(loc_desc)
      # print(location.name)
      # print(location.description())
      # print("People Present:")
      # for agent in location.people:
      #   # print(f"\t{agent.name}")
      #   print(agent.description())
    print(yaml.dump(location_descriptions))    

  def get_agent(self, agent_name : str):
      for agent in self.agents:
        if agent.name == agent_name:
           return agent
    
      print("Error: agent not found")
      return None

  def create_world_from_names(self, location_name_list, agent_description_list, model_object = None):
    """
    location_name_list = list of strings
    agent_description_list = list of dictionaries with keys "name", "status", and "traits" attributes
    """

    self.model = model_object

    self.clear()

    for loc_name in location_name_list:
      new_location = Location(loc_name)
      self.add_location(new_location)

    for agent_dict in agent_description_list:
      name = agent_dict["name"]
      traits = agent_dict["traits"]
      status = agent_dict["status"]
      if "location" in agent_dict:
        loc_name = agent_dict["location"]
        # print("agent in location", loc_name)
        for location in self.locations:
          if location.name == loc_name:
            # print("adding agent to location", loc_name)
            agent = SimpleAgent(name, traits, status, SimpleMemory(), self.model, location)
            self.add_agent(agent)
            location.add_person(agent)
            break
      else:
        raise ValueError(f"Agent {name} has no location")
        agent = SimpleAgent(name, traits, status, SimpleMemory())
        self.add_agent(agent)

      # agent = SimpleAgent(name, traits, status, SimpleMemory())
      # self.add_agent(agent)
      # if "location" in agent_dict:
      #   loc_name = agent_dict["location"]
      #   for location in self.locations:
      #     if location.name == loc_name:
      #       # self.place_agent(agent, location)
      #       location.add_person(agent)
      #       # agent.location = location
      #       agent.set_location(location)
      #       # break

  def add_agent(self, agent):
    self.agents.append(agent)
  def add_location(self, location):
    self.locations.append(location)

  def read_adjacency_list(self,list_of_adjacencies):
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
      # self.set_adjacent(location1, location2)
      location1.add_adjacent_location(location2)
      location2.add_adjacent_location(location1)

  # def set_adjacent(loc1, loc2):
  #   loc1.add_adjacent_location(loc2)
  #   loc2.add_adjacent_location(loc1)

  def place_agent(self, agent, location):
    """ Use to initialize placement with no error checking """
    # location.add_person(agent)
    # agent.location = location

  # def move_agent(self, agent, destination):
  #   if not(start_loc.contains_person(agent)):
  #     raise ValueError(f"{agent.name} is not in {start_loc.name}")
  #   if not(end_loc in start_loc.adjacent_locations):
  #     raise ValueError(f"{agent.name} cannot move from {start_loc.name} to {end_loc.name}")
  #   end_loc.add_person(agent)
  #   start_loc.remove_person(agent)
  #   agent.location = end_loc

  ## Functions that agents will be able to call
  ## Take *STRING INPUTS ONLY
  def move(self, agent_name: str, dest_name: str) -> str:

      for person in self.agents:
          if person.name == agent_name:
              agent = person
              break
      else:
          return f"Agent {agent_name} not found."

      for place in self.locations:
          if place.name == dest_name:
              dest = place
              break
      else:
          return f"Destination {dest_name} not found."

      # agent = self.agents[agent_name]
      # dest  = self.locations[dest_name]
      start = agent.location
      if dest not in start.adjacent_locations:
          return f"{dest_name} is not reachable from {start.name}."
      start.remove_person(agent)
      dest.add_person(agent)
      agent.location = dest
      return f"{agent_name} moved from {start.name} to {dest.name}."

  def describe_room(self, agent_name: str) -> str:
      # loc = self.agents[agent_name].location
      for place in self.locations:
          if place.contains_person(agent_name):
              loc = place
              break
      else:
          return f"Agent {agent_name} not found."
      names = ', '.join(p.name for p in loc.people if p.name != agent_name) or "no-one"
      return f"You are in the {loc.name}. People here: {names}."
