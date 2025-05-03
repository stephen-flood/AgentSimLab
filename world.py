from agent import SimpleAgent, SimpleMemory
from typing import List
import yaml 

# Handcoded (with occasional LLM assistance)

class Location:
  name : str
  people : List[SimpleAgent]
  general_description : str
  # things : List[Thing]
  # adjacent_locations : List[Location]

  # ## Create/modify location
  # def __init__(self, name, people_present=None, adjacent_locations=None) -> None:
  #   if people_present is None:
  #     people_present = []
  #   if adjacent_locations is None:
  #     adjacent_locations = []
  #   self.general_description = ""
  #   self.name = name
  #   self.people = people_present
  #   self.adjacent_locations = adjacent_locations
  def __init__(self, name, **kwargs) -> None:
    self.people = kwargs["agents"] if "agents" in kwargs else []
    self.adjacent_locations = kwargs["adjacent_locations"] if "adjacent_locations" in kwargs else []
    self.general_description = ""
    self.name = name
    # Include pointer back to world.  Needed for agent tools to be able to call world functions like "move"
    self.world = kwargs["world"] if "world" in kwargs else []

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
      # adjacent_descriptions += f"{loc.name}:{loc.general_description},"
      adjacent_descriptions.append(loc.name)

    person_descriptions=[]
    for person in self.people:
      person_descriptions.append( person.description_dictionary() )

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
        self.add_location(new_location)

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
        persona = agent_dict["persona"]
        status = agent_dict["status"]
        if "location" in agent_dict:
          loc_name = agent_dict["location"]
          # print("agent in location", loc_name)
          for location in self.locations:
            if location.name == loc_name:
              # print("adding agent to location", loc_name)
              agent = SimpleAgent(name, 
                                  traits=persona, 
                                  status=status, 
                                  memory=SimpleMemory(), 
                                  model=self.model,
                                  location=location)
              self.add_agent(agent)
              location.add_person(agent)
              break
        else:
          print(f"Agent {name} has no location")
          agent = SimpleAgent(name, persona, status, SimpleMemory())
          self.add_agent(agent)
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


  def get_agent(self, agent_name : str):
      for agent in self.agents:
        if agent.name == agent_name:
           return agent
      print("Error: agent not found")
      return None
  
  def get_location(self, location_name : str):
      for location in self.locations:
         if location.name == location_name:
            return location
      print("Location not found")
      return f"Error: Location {location_name} not found"
  

  def set_adjacent(self,location1 : Location, location2 : Location):
    if location1 not in self.locations | location2 not in self.locations:
        print(f"Error: {location1.name} or {location2.name} not in World")
        return
    location1.add_adjacent_location(location2)
    location2.add_adjacent_location(location1)
    return 

  def set_not_adjacent(self,location1 : Location, location2 : Location):
    if location1 not in self.locations | location2 not in self.locations:
        print(f"Error: {location1.name} or {location2.name} not in World")
        return
    location1.set_non_adjacent(location2)
    location2.set_non_adjacent(location1)
    return 

  def add_agent(self, agent):
    self.agents.append(agent)
  def add_location(self, location):
    self.locations.append(location)

  ## TODO: decide exactly how to move agents based on tool implemntation at simulation level
  ## NOTE:  
  ##    1. Agent name should NOT be set by the agent, but by the simulation
  ##    2. That is inconsistent with the `easy tool calling` implemented in model.py, 
  ##       or at least with defining a SINGLE function for all agents and using model.py approach

  # def move_agent(self, agent, destination):
  #   if not(start_loc.contains_person(agent)):
  #     raise ValueError(f"{agent.name} is not in {start_loc.name}")
  #   if not(end_loc in start_loc.adjacent_locations):
  #     raise ValueError(f"{agent.name} cannot move from {start_loc.name} to {end_loc.name}")
  #   end_loc.add_person(agent)
  #   start_loc.remove_person(agent)
  #   agent.location = end_loc

  #############################################
  #####  MODIFY AS NEEDED FOR TOOL USE    #####
  #############################################

  ## To Implement a tool that modifies the world
  # 1. Define a WORLD method **HERE**  
  #     e.g. define `world.move(dest_name, agent))`` as below
  # 2. Define a STANDALONE WRAPPER where you *NEED THE TOOL** 
  #     e.g. `define move_tool(dest_name, agent)` 
  #     ```def move_tool(destination: str, agent: SimpleAgent) -> str:
  #           # assumes each Location has a back-reference `world`
  #           world = agent.location.world
  #           world.move_agent(agent, destination)
  #           return f"{agent.name} moved to {destination}."
  #     ```
  # 3. Create the ATTRIBUTE dictionary, but **DO_NOT** include the agent:
  #      ```
  #      model.register_tool(
  #          func=move_tool,
  #          description="Move the calling agent to an adjacent location.",
  #          parameters={
  #              "destination": {
  #                  "type": "string",
  #                  "description": "Name of the adjacent location to move to."
  #              }
  #          }
  #      )```
  # 4. Process the response with the AGENT version apply_agent_tool
  #        ```
  #        response = agent.generate_action(
  #            tools=[model.get_tool("move_tool")],
  #            instruction="Choose an adjacent location and call move_tool with the destination.",
  #        )
  #        agent.apply_agent_tool(response)
  #        ```


  ## Functions that agents will be able to call
  ## Take *STRING INPUTS ONLY
  # def move(self, dest_name: str, agent: SimpleAgent ) -> str:
  def move(self, **kwargs) -> str:
      dest_name = kwargs["dest_name"]
      agent = kwargs["agent"] 
      # ONLY advertise dest_name
      # agent will be set later when the tool is called by the appropriate agent

      dest = self.get_location(dest_name)
      if dest is str:
         # No location was found, so return the error message.
         print(f"Warning: no location of name {dest} found")
         return False

      start = agent.get_location()
      if dest not in start.adjacent_locations:
          print(f"Warning: {dest_name} is not reachable from {start.name}.")
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


class Simulation:
   
  def __init__(self, agent_description_list, location_list,  adjacency_list, freemodel ):
      self.world = World()
      self.world.create_world_from_names( location_list , agent_description_list , freemodel )
      self.world.read_adjacency_list(adjacency_list)

  def forward():
      pass 
   