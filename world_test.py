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



def get_room_description(**kwargs) ->str:
    required_args=["agent"]
    if any( arg not in kwargs for arg in required_args):
        print(f"Error: missing argument in {kwargs}.  Required arguments: {required_args}.")
        return False
    else: 
        agent = kwargs["agent"]
    print(f"{agent.name} looks around.")
    return agent.get_location().description()
describe = freemodel.register_tool(
    func=get_room_description,
    description="You look around your current room to see who or what is in it, and what is nearby.",
    parameters={
    }
)

# (1) The function you will run to perform the action. 
#       Note: You do NOT need to advertise all arguments to the LLM
def move_tool(**kwargs) -> str:
    # Check required inputs
    required_args = ["destination","agent","world"]
    if any( arg not in kwargs for arg in required_args):
        print(f"Error: missing argument in {kwargs}.  Required arguments: {required_args}.")
        return False
    else:
        destination = kwargs["destination"] 
        agent = kwargs["agent"]
        world = kwargs["world"]
    result = world.move(agent=agent, dest_name=destination)
    # Check whether move was valid or invalid
    if "Warning" in result:
        print(result)
        return result
    else:
        print( f"{agent.name} moved to {destination}.")
        return result
# (2) Register tool with the model
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
# tools=[move]
tools=[move,describe]

## Prompt (trying) to trigger `movement`
# observation = "I am falling asleep. I need coffee to able to keep going, but there is no coffee here. "
## Prompt (trying) to trigger `describe`
observation = "I am falling asleep. Is there any coffee in in my mug?"

susan = world.get_agent("Susan")
susan.add_memory("Observation: " + str(observation))

plan = susan.generate_plan()
susan.add_memory("Plan: " + str(plan))

# response = susan.generate_speech(tools=tools)
# print("speech:", response)
# observation += response

# (3) Supply tool objects during agent.generate_action
act = susan.generate_action(
    tools=tools,
    memory = observation,
)
if act.function_calls is not None:
    for call in act.function_calls:
        intended_action = f"Calling {call.name} with arguments {call.args}"
        susan.add_memory("Attempting Action: " + intended_action)

# (4) Use model.apply_tool.  Remember to supply all "hidden" arguments, not advertised to the model
observe = freemodel.apply_tool(act, world=world, agent=susan)
for result in observe:
    susan.add_memory("Observation: " + str(result[0]))

# world.print()

print(susan.description())

# ## Prompt (trying) to trigger `movement`
# # observation = "I am falling asleep. I need coffee to able to keep going, but there is no coffee here. "
# ## Prompt (trying) to trigger `describe`
# observation = "I am falling asleep. Is there any coffee here?"

# susan = world.get_agent("Susan")

# plan = susan.generate_plan(memory=observation,tools=tools)
# print("plan", plan)

# memory = memory + "\n" + plan

# response = susan.generate_speech(memory=observation, tools=tools)
# print("speech:", response)
# observation += response

# # (3) Supply tool objects during agent.generate_action
# response = susan.generate_action(
#     tools=tools,
#     memory = observation,
# )
# # (4) Use model.apply_tool.  Remember to supply all "hidden" arguments, not advertised to the model
# response = freemodel.apply_tool(response, world=world, agent=susan)

# world.print()

# # response = susan.respond("You are a tool calling agent simulating Susan.  You can call get_current_room_description and move_to_location.  Wander amlessly thorugh the building.", tools=tools)
# # print(response)

# # print("Moving Susan")
# # susan = world.get_agent("Susan")
# # for i in range(10):
# #     response = susan.respond("Wander amlessly thorugh the building.  Use tool calling.", tools=tools)
# #     print(response)
# #     print(susan.location.name)
# # print(tools)