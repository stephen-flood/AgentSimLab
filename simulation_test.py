# main.py
from model  import GeminiModel, HTTPChatModel
from world  import World
from agent  import SimpleAgent

## If the library is a folder, import FILENAMES
# from OpenSimLab import world, agent, model

## If you want specific objects from each files, do this
# from OpenSimLab.world import World
# from OpenSimLab.model import GeminiModel
# from OpenSimLab.agent import SimpleAgent

# (hacked together examples, cleaned up by o3)

# ---------- 1.  Create model instance ---------------------------------------
# model = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
# model = GeminiModel("gemini-2.0-flash", 15, 1000)
# model = GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
# model = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")
model = HTTPChatModel("gemma3:12b", native_tool=False)

# ---------- 2.  Build the world ---------------------------------------------
rooms   = ["living", "kitchen", "bathroom", "bedroom", "office", "garden"]
edges   = [("living","kitchen"), ("living","bathroom"), ("living","bedroom"),
           ("living","office"), ("living","garden"), ("kitchen","garden")]
agents  = [
    # {"name":"Susan",
    #     "observation" : "I need to get some coffee.",
    #     "status":"working",
    #     "persona":"tired" , 
    #     "location":"living"},
    # {"name":"Bob"  ,
    #     "observation" : "I need to go outside."
    #     "status":"cooking",
    #     "persona":"lazy"  , 
        # "location":"kitchen"},
    {"name":"Timmy",
        "observation":"I am having a great time.",
        "status":"playing",
        "persona":"active", 
        "persona":"active", 
        "location":"garden"},
    {"name":"Catherine",
        "goal":"I need to TELL Timmy to do his homework. I need to find the LOCATION that contains Timmy. I should first CHECK if my CURRENT LOCATION.  If not I should MOVE to a new LOCATION to check if he is there. REMEMBER to alternate looking and moving.",
        "status":"curious",
        # "persona":"focused", 
        # "persona":"explorer",
        "location":"office"},
]
world = World(location_names=rooms, room_edges=edges,
              agent_descriptions=agents, model=model)

# ---------- 3.  Register imported functions as Gemini tools -----------------

def get_room_description(**kwargs) ->str:
    try:
        room_desc = kwargs["agent"].get_location().description()
        print(f"{kwargs["agent"].name} looks around and sees {room_desc}.")
        return kwargs["agent"].get_location().description()
    except:
        err_msg = f"Error: missing argument for get_room_description in {kwargs}"
        print(err_msg)
        return err_msg
describe_tool = model.register_tool(
    func        = get_room_description,          # pure function from tools.py
    description = "Look around the current room",
    parameters  = {})

def move(**kwargs) -> str:
    try:
        result = kwargs["world"].move(agent=kwargs["agent"], dest_name=kwargs["destination"] )
        # TODO: Handle invalid arguments (e.g. inaccessible destination?)
        return result
    except:
        err_msg = f"Error: missing argument for move_tool in {kwargs}"
        print(err_msg)
        return err_msg
move_tool = model.register_tool(
    func        = move,             # pure function from tools.py
    description = "Move to an adjacent room",
    parameters  = {
        "destination": {
            "type": "string",
            "description": "The name of the room to move to."
        }
    })
def speak(**kwargs) -> str:
    try:
        return kwargs["agent"].generate_speech(interlocutor_name=kwargs["interlocutor_name"])
    except:
        print("ERROR SPEAKING")
        return "ERROR SPEAKING"
speak_tool = model.register_tool(
    func = speak,
    description="Talk to a named individual",
    parameters={
        "interlocutor_name" :{
            "type" : "string",
            "description" : "The name of the individual you are talking to."
        }
    }
)
TOOLS = [move_tool, describe_tool]              

# ---------- 4.  Tiny Simulation with step() ---------------------------------
class Simulation:
    """Between option 1 and 5: has step() *and* run(num_steps)."""
    def __init__(self, world, tools):
        self.world = world
        self.tools = tools
        self.t     = 0

    def step(self):
        """One synchronous tick over all agents."""
        for agent in self.world.agents:         # assumes iterable interface
            print(f"--------- {agent.name} at step {self.t} ---------")
            # print(agent.description())
            if self.t == 0:
                plan = agent.generate_plan(tools=self.tools)
                # agent.add_memory("Plan: " + plan)
                # print("Plan: " + plan)
            else:
                # plan = agent.generate_plan(context = "Summarize what you have attempted, then formulate your plan. Always look for something NEW to try.")
                plan = agent.generate_plan(context = "Start by summarizing what you have already done. Look for new arguments for tools you have already", tools=self.tools)
            # plan = agent.generate_plan(tools=self.tools)
            agent.add_memory("Plan: " + plan)
            print("Plan: " + plan)

            act  = agent.generate_action(tools=self.tools)

            # OPTIONAL: Describe the tools to be called BEFORE calling them
            tool_call_list = model._iter_tool_calls(act)
            if tool_call_list is not None:
                for call in tool_call_list:
                    intended_action = f"Calling {call[0]} with arguments {call[1]}"
                    agent.add_memory("Attempting Action: " + intended_action)
                    print("Attempting Action: " + intended_action)
            else: 
                print("No tools called")

            # ACTUALLY call the tools:
            observe = self.world.model.apply_tool(act, world=self.world, agent=agent)
            for result in observe:
                agent.add_memory("Observation: " + str(result[0]))
                print("Observation: " + str(result[0]))

        self.t += 1

    def run(self, steps=1):
        for _ in range(steps):
            self.step()

# ---------- 5.  Run the demo -------------------------------------------------
sim = Simulation(world, TOOLS)
sim.run(steps=10)
# world.print()
