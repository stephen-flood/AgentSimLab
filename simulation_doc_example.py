from model  import GeminiModel
from world  import World
from agent  import SimpleAgent

# 1️⃣  Create the shared language model
llm = GeminiModel("gemini-2.0-flash-lite", 25, 1000)

# 2️⃣  Describe the physical space
rooms = ["kitchen", "living", "garden"]
edges = [("kitchen","living"), ("living","garden")]

# 3️⃣  Describe the inhabitants
agents = [
    {"name":"Alice", "persona":"curious", "status":"reading", "location":"kitchen"},
    {"name":"Bob"  , "persona":"sleepy" , "status":"napping", "location":"living"},
]

# 4️⃣  Build the world object
world = World(location_names=rooms,
              room_edges=edges,
              agent_descriptions=agents,
              model=llm)

# 5️⃣  (Optional) expose world‑mutating functions as Gemini tools
def move(**kwargs):
    return kwargs["world"].move(
        agent=kwargs["agent"],
        dest_name=kwargs["destination"]
    )

move_tool = llm.register_tool(
    func=move,
    description="Move to an adjacent room",
    parameters={"destination":{"type":"string",
                               "description":"Name of the room to move to."}}
)

TOOLS = [move_tool]      # single source‑of‑truth list

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
            for text, call in llm.apply_tool(action, world=self.world, agent=ag):
                ag.add_memory("Observation: " + text)
                print(text)

        self.t += 1

    def run(self, steps=1):
        for _ in range(steps):
            self.step()

Simulation(world, TOOLS).run(steps=3)
world.print()