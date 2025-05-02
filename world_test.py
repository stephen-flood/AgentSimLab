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
world = World()
world.clear()
world.create_world_from_names( location_list , agent_description_list , freemodel )
world.read_adjacency_list(adjacency_list)
world.print()

"""## Very Simple Simulation (users move between locations)

"""




# Define function and register tool
# NOTE: you need to use the AGENT's version of `apply_tool_calls`
# This will *INSERT* the correct agent parameter (hidden from the LLM)
def move_to_location(dest: str, agent: SimpleAgent) -> str:
    return agent.move(dest)
move_tool = freemodel.register_tool(
    func=move_to_location,
    description="Moves you to an adjacent room.",
    parameters={
        "dest":       {"type": "string", "description": "An adjacent room."},
    }
)

def get_current_room_description(agent : SimpleAgent) -> str:
    return agent.location.description()
describe_tool = freemodel.register_tool(
    func=get_current_room_description,
    description="Describes your current room and who is there.",
    parameters={
    }
)
tools = [move_tool, describe_tool]


susan = world.get_agent("Susan")
# response = susan.respond("Wander amlessly thorugh the building.", tools=tools)
response = susan.generate_speech(instruction="Wander amlessly thorugh the building.", tools=tools)
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