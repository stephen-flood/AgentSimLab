# OpenSimLab 

- Written spring of 2025. 
- Prompting approach inspired by an approach I found (somewhere), where all the prompts were things like
        Persona: friendly
        Instruction: run fast
    Thus the agent approach just takes in an arbitrary dictionary of attributes, and the prompt is the `yaml.dump` of the values.
- *Documentation written by o3, 5/3/25*
    - Has not been double checked, use with care. 
    - It was a huge pain to get it to output the markdown because all the code snippets broke their chat interface
    - Eventually, the following prompt worked: 
        Perfect.  Now I need this EXACT content in a file that I can download.

        The file I want is UNUSUAL.  
        TEXT content should be rendered in markdown
        CODE should be placed between <CODE> and </CODE> delimiters.

        NEVER use a triple backtick when outputting your answer 

# Quick‑Start Guide

A lightweight, hack‑friendly sandbox for multi‑agent simulations built on top of Google Gemini.  
This guide shows **(1) how to use `GeminiModel` for basic chat / multimodal inference** and **(2) how to build and run simple simulations**.  
It is aimed at undergraduates encountering LLMs for the first time, so every section comes with runnable code snippets.

---

## 0. Prerequisites
```
pip install google-genai pyyaml
```

1. **Get an API key** for Gemini and either  
   *save it in an environment variable*  
```
export GEMINI_API_KEY="paste-key-here"
```  
   or put it in a text file named `gemini.api_key` next to your notebooks / scripts.

2. Clone or download the project files (`model.py`, `agent.py`, `world.py`, …).

---

## 1. Your first chat with `GeminiModel`
```
from model import GeminiModel

# make a model instance – arguments: (model_name, per_min_limit, daily_limit)
model = GeminiModel("gemini-2.0-flash-lite", 25, 1_000)

response = model.generate_content(
    user_prompt="Write a haiku about debugging code."
)
print(response.text)
```

### 1.1 Rate‑limits handled for you  
`GeminiModel` wraps each request with a `RateLimitTracker` so you don’t have to sprinkle
`time.sleep()` in your notebooks. Feel free to open `model.py` and tweak the policy.

### 1.2 Multimodal queries (images / video / URLs)
```
resp = model.multimodal_query(
    user_prompt="What flower is this?",
    attachment_names=["https://upload.wikimedia.org/wikipedia/commons/4/49/Rose.JPG"]
)
print(resp.text)
```

---

## 2. Calling Python from the LLM (tool use)
```
def cube(x: int) -> int:
    return x**3

cube_tool = model.register_tool(
    func=cube,
    description="Returns the cube of an integer.",
    parameters={"x": {"type": "integer",
                      "description": "Number to raise to the third power."}}
)

# The tool object is what you pass to Gemini
reply = model.generate_content(
    user_prompt="What is 7 cubed?  Use the tool if you need it.",
    tools=[cube_tool]
)
model.print_response(reply)           # either plain text or a function call
print(model.apply_tool(reply))        # -> [('343', <FunctionCall …>)]
```

---

## 3. Building blocks for simulations

| Class | What it represents | Where to look |
|-------|--------------------|---------------|
| `SimpleAgent` | a thinking, talking entity with attributes, memory and a `GeminiModel` inside | `agent.py` |
| `Location`    | a place that can hold agents and be connected to other places           | `world.py` |
| `World`       | the network of `Location`s plus convenience helpers                     | `world.py` |
| `Simulation`  | the game‑loop that advances the world step by step                      | *you write this* or start from `simulation_test.py` |

All four are plain Python – open the files and hack away!

---

## 4. A minimal working simulation
```
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
            print(f"\\n--- {ag.name} (t={self.t}) ---")

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
```

Copy–paste the snippet into a notebook – you should see the agents talk, decide, and move.

---

## 5. Next steps

* **Tune prompts** – Each `SimpleAgent` builds its prompt from its internal
  attributes *plus* whatever you pass to `generate_plan` / `generate_action` / `generate_speech`.
* **Add new tools** – any pure Python function can become a tool.  
  Remember to return plain text (Gemini currently ignores complex objects).
* **Extend the world** – locations can store objects, descriptions, or even
  numeric state; just add fields to `Location` and teach agents via tools.

Happy hacking!