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
    try:
        print(f"{kwargs["agent"].name} looks around.")
        return kwargs["agent"].get_location().description()
    except:
        err_msg = f"Error: missing argument for get_room_description in {kwargs}"
        print(err_msg)
        return err_msg
    # required_args=["agent"]
    # if any( arg not in kwargs for arg in required_args):
    #     err_msg = f"Error: missing argument in {kwargs}.  Required arguments: {required_args}."
    #     print(err_msg)
    #     return err_msg
    # print(f"{kwargs["agent"].name} looks around.")
    # return kwargs["agent"].get_location().description()
describe = freemodel.register_tool(
    func=get_room_description,
    description="You look around your current room to see who or what is in it, and what is nearby.",
    parameters={
    }
)

# (1) The function you will run to perform the action. 
#       Note: You do NOT need to advertise all arguments to the LLM
def move_tool(**kwargs) -> str:
    try:
        result = kwargs["world"].move(agent=kwargs["agent"], dest_name=kwargs["destination"] )
        # TODO: Handle invalid arguments (e.g. inaccessible destination?)
        return result
    except:
        err_msg = f"Error: missing argument for move_tool in {kwargs}"
        print(err_msg)
        return err_msg
    # Check required inputs
    # required_args = ["destination","agent","world"]
    # if any( arg not in kwargs for arg in required_args):
    #     err_msg = f"Error: missing argument in {kwargs}.  Required arguments: {required_args}."
    #     print(err_msg)
    #     return err_msg
    # result = kwargs["world"].move(agent=kwargs["agent"], dest_name=kwargs["destination"] )
    # # TODO: Handle invalid arguments (e.g. inaccessible destination?)
    # return result
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

## TODO: Tool for "SPEAKING" to another person in the current room.


## Prompt (trying) to trigger `movement`
observation = "I am falling asleep. I need coffee to able to keep going, but there is no coffee here. "
## Prompt (trying) to trigger `describe`
# observation = "I am falling asleep. Is there any coffee in in my mug?"

susan = world.get_agent("Susan")
susan.add_memory("Observation: " + str(observation))

plan = susan.generate_plan()
susan.add_memory("Plan: " + str(plan))

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
print(susan.description())
