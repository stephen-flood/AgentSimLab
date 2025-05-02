from model import GeminiModel, RateLimitTracker
from agent import SimpleAgent, SimpleMemory
from world import World

"""## World Class Test"""
freemodel = GeminiModel("gemini-2.0-flash-lite", 30, 1000)

location_list = ["living", "kitchen", "bathroom", "bedroom","office","garden"]
agent_description_list = [
  {"name": "Susan", "status": "working", "persona": "tired", "location": "living"},
  {"name": "Bob", "status": "cooking", "persona": "lazy", "location": "kitchen"},
  {"name": "Timmy", "status": "playing", "persona": "active", "location": "garden"},
  {"name": "Catherine", "status": "curious", "persona": "focused", "location": "office"},
]
adjacency_list = [("living","kitchen"),
                  ("living","bathroom"),
                  ("living","bedroom"),
                  ("living","office"),
                  ("living","garden"),
                  ("kitchen","garden"),]
# world = World()
# world.create_world_from_names( location_list , agent_description_list , freemodel )
world = World( location_names = location_list, room_edges=adjacency_list , agent_descriptions = agent_description_list , model = freemodel )
# world.read_adjacency_list(adjacency_list)
world.print()

"""## Very Simple Simulation (users move between locations)

"""




# Define function and register tool
# NOTE: you need to use the AGENT's version of `apply_tool_calls`
# This will *INSERT* the correct agent parameter (hidden from the LLM)
# def move_to_location(dest: str, agent: SimpleAgent) -> str:
#     return agent.move(dest)
# move_tool = freemodel.register_tool(
#     func=move_to_location,
#     description="Moves you to an adjacent room.",
#     parameters={
#         "dest":       {"type": "string", "description": "An adjacent room."},
#     }
# )

# def get_current_room_description(agent : SimpleAgent) -> str:
#     return agent.location.description()
# describe_tool = freemodel.register_tool(
#     func=get_current_room_description,
#     description="Describes your current room and who is there.",
#     parameters={
#     }
# )
# tools = [move_tool, describe_tool]

def move_tool(**kwargs) -> str:
    destination = kwargs["destination"] 
    agent = kwargs["agent"]

    # assumes each Location has a back-reference `world`
    world = agent.get_location().world
    world.move(agent=agent, dest_name=destination)
    return f"{agent.name} moved to {destination}."

# 2) Registration with the model
move = freemodel.register_tool(
    func=move_tool,
    description="This function moves you from your current room to an adjacent room.",
    parameters={
        "destination": {
            "type": "string",
            "description": "The name of the room that you want to move to."
        }
    }
)
tools=[move]

memory = "I am falling asleep. I need coffee to able to keep going, but there is no coffee here. "
susan = world.get_agent("Susan")
# response = susan.respond("Wander amlessly thorugh the building.", tools=tools)
response = susan.generate_speech(memory=memory, tools=tools)
print(response)
memory += response

# 3) Use in agent.generate_action
response = susan.generate_action(
    # tools=[freemodel.get_tool("move_tool")],
    tools=tools,
    memory = memory,
    # system = "You are a tool calling model.  You MUST call one of your tools.",
    # instruction="Move to the *kitchen*. That means call `move_tool` with argument `kitchen`.",
)
print(response)
response = susan.apply_agent_tool(response)
# memory += response
print(response)

# response = susan.respond("You are a tool calling agent simulating Susan.  You can call get_current_room_description and move_to_location.  Wander amlessly thorugh the building.", tools=tools)
# print(response)

# print("Moving Susan")
# susan = world.get_agent("Susan")
# for i in range(10):
#     response = susan.respond("Wander amlessly thorugh the building.  Use tool calling.", tools=tools)
#     print(response)
#     print(susan.location.name)
# print(tools)