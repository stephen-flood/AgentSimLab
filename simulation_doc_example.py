from model  import GeminiModel, HTTPChatModel, HFTransformersModel
from world  import World
from agent  import SimpleAgent, SimpleModel, SelfCompressingMemory

# 1️⃣  Create the shared language model
## Google API
# llm = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
# llm = GeminiModel("gemini-2.0-flash", 15, 1000)
# llm = GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
## Ollama
# llm = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")
llm = HTTPChatModel("gemma3:12b", native_tool=False)
## HF transformers
# llm = HFTransformersModel( "allenai/OLMo-2-0425-1B-Instruct", native_tool=False, )


# 2️⃣  Describe the physical space
rooms = ["kitchen", "living", "garden"]
edges = [("kitchen","living"), ("living","garden")]

# 3️⃣  Describe the inhabitants
agents = [
    {"name":"Alice", 
        "persona":"curious", 
        "status":"reading", 
        "location":"kitchen",
        # "memory" :  SimpleMemory(),
        # "memory" :  SelfCompressingMemory(500,llm), # Memory limited to given number of CHARACTERS
        "memory" :  SelfCompressingMemory(1000,llm), # Memory limited to given number of CHARACTERS
    },
    {"name":"Bob"  , 
        "persona":"sleepy" , 
        "status":"napping", 
        "location":"living",
        # "memory" :  SimpleMemory(),
        # "memory" :  SelfCompressingMemory(500,llm),# Memory limited to given number of CHARACTERS
        "memory" :  SelfCompressingMemory(1000,llm), # Memory limited to given number of CHARACTERS
    },
]

# 4️⃣  Build the world object
world = World(location_names=rooms,
              room_edges=edges,
              agent_descriptions=agents,
              model=llm)

# 5️⃣  (Optional) expose world‑mutating functions as Gemini tools
def move(**kwargs):
    try:
        return kwargs["world"].move(
            agent=kwargs["agent"],
            dest_name=kwargs["destination"]
        )
    except:
        return "Invalid move command with arguments {args}.".format(args=kwargs)
move_tool = llm.register_tool(
    func=move,
    description="Move to an adjacent room",
    parameters={"destination":{"type":"string",
                               "description":"Name of the room to move to."}}
)


def get_room_description(**kwargs) ->str:
    try:
        room_desc = kwargs["agent"].get_location().description()
        print(f"{kwargs["agent"].name} looks around and sees {room_desc}.")
        return kwargs["agent"].get_location().description()
    except:
        err_msg = f"Error: missing argument for get_room_description in {kwargs}"
        print(err_msg)
        return err_msg
look_tool = llm.register_tool(
    func        = get_room_description,          # pure function from tools.py
    description = "Look around the current room.",
    parameters  = {})

TOOLS = [move_tool, look_tool]      # single source‑of‑truth list

# 6️⃣  Write the game loop — *now with explicit state updates*
class Simulation:
    def __init__(self, world, tools):
        self.world = world
        self.tools = tools
        self.t = 0

    def step(self):
        for ag in self.world.agents:
            print(f"\n--- {ag.name} (t={self.t}) ---")

            # ➊ Think: make a plan **and remember it**
            plan = ag.generate_plan(tools=self.tools)
            ag.add_memory("Plan: " + plan)  # manual state update
            print(plan)

            # ➋ Act: pass the plan back in so the agent can reference it
            action = ag.generate_action(tools=self.tools, plan=plan)

            # ➌ Observe: execute any tool calls and store what happened
            # OPTIONAL: Describe and remember the tools you plan to call
            tool_call_list = llm._iter_tool_calls(action)
            if tool_call_list is not None:
                for call in tool_call_list:
                    intended_action = f"Calling {call[0]} with arguments {call[1]}"
                    ag.add_memory("Attempting Action: " + intended_action)
                    print("Attempting Action: " + intended_action)
            else: 
                print("No tools called")

            # ACTUALLY call the tools
            observe = llm.apply_tool(action, world=self.world, agent=ag)
            for result in observe:
                ag.add_memory("Observation:\n" + str(result[0]))
                # print("Observation:\n" + str(result[0]))

        self.t += 1

    def run(self, steps=1):
        for _ in range(steps):
            self.step()

# Simulation(world, TOOLS).run(steps=3)
Simulation(world, TOOLS).run(steps=30)

world.print()