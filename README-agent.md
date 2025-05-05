# `agent.py` – Deep Dive Reference
A practical look at the **memory‑driven, tool‑aware agents** that power OpenSimLab.  After reading you should be able to  
* extend `SimpleAgent` with custom state or behaviour,  
* swap out the memory model, and  
* wire multiple agents together as in `agent_test.py`.

---

## 1. Design goals
| Goal | How it shows up in code |
|------|------------------------|
| *Minimal surface area* | Only **three public helper classes**: `SimpleMemory`, `Location`, `SimpleAgent`. |
| *Readable prompts* | All agent → LLM calls are YAML dumps of the agent’s own state. No hidden metaprompts. |
| *Composable tools* | Tools live in the **model’s registry**; agents just pass `[tool1, tool2]` to `generate_action`. |
| *Explicit side‑effects* | `generate_plan` / `generate_action` *do not* mutate state; you choose when to `.add_memory()` or feed outputs forward. |

---

## 2. `SimpleMemory`
<CODE>
mem = SimpleMemory([
    "I bought a phone six weeks ago",
    "My phone is an Android"
])
print(str(mem))  # YAML list or "N/A" if empty
mem.add_memory("I am hungry")
latest_two = mem.retrieve_recent_memories(2)
</CODE>

*Internally just a `List[str]`* so you can slice, filter, or pickle it easily.

### Key points
* `__str__` returns a YAML dump → perfectly readable when embedded into prompts.  
* `retrieve_recent_memories(n)` returns **all** memories when `n < 0` – handy for a full context reset.

---

## 3. `SimpleAgent`
### 3.1 Constructor
<CODE>
agent = SimpleAgent(
    "SupportAgent",
    persona="calm, helpful",
    status="waiting to assist",
    memory=SimpleMemory(),
    model=freemodel,
)
</CODE>

* **`name` (str)** – required & unique.  
* **Special kwargs**: `model`, `tools` are handled up‑front.  
* **Everything else** is dumped into `attribute_dictionary` and must implement `__str__`.

### 3.2 Prompt assembly (`generate_content` under the hood)
1. `attribute_dictionary` ➕ caller’s `kwargs` ⇒ merged dict. Caller wins on key collisions.  
2. Convert to strings (`__str__` called automatically).  
3. Feed into `Prompt(**dict).generate()`.  
4. Pass along *optional* `model_parameters` such as `tools=[…]`, `temperature=0.2`, etc.

### 3.3 Behaviour helpers
| Method | Instruction auto‑injected | Typical usage |
|--------|--------------------------|---------------|
| `generate_plan()` | *“First, identify what **name** would do. Then make a very short plan …”* | Kick off a turn; store the plan if you like. |
| `generate_action()` | *“What would **name** do?”* + system/tool hint | Tool‑calling step; run `model.apply_tool()` yourself. |
| `generate_speech()` | *“What would **name** say (to X)?”* | Free‑text chat bubble. |

All three return a **raw Gemini response** (string or object). **Nothing is persisted** unless you call `.add_memory()`.

### 3.4 Memory convenience
<CODE>
agent.add_memory("Customer says: I dropped my phone")
</CODE>
Raises if agent has no `memory` attribute – keeps state errors obvious.

### 3.5 Location helpers (optional)
If you attach a `Location` object (you supply the class!), `get_location()` and `set_location()` offer a minimal API without hard‑coding world logic.

---

## 4. End‑to‑end conversation (`agent_test.py` excerpt)
<CODE>
freemodel = GeminiModel()
support = SimpleAgent("SupportAgent", persona="calm", memory=SimpleMemory(), model=freemodel)
customer = SimpleAgent("Customer", persona="impatient", memory=SimpleMemory(), model=freemodel)

observation = 'Customer says: "My phone keeps rebooting!"'
customer.add_memory(observation)
support.add_memory(observation)

for _ in range(3):
    # Support speaks
    reply = support.generate_speech(observation=observation)
    observation = "Support says: " + reply
    customer.add_memory(observation)
    support.add_memory(observation)
    print(observation)

    # Support tries a tool
    action_resp = support.generate_action(model_parameters={"tools":[refund_tool]})
    result = freemodel.apply_tool(action_resp)
    print(result)
</CODE>

**Takeaways:**  
* Loop orchestrates *who* calls *which* method – agents never auto‑switch turns.  
* Tool calls are explicit (`model_parameters={"tools": [...]}`).  
* Conversation persists by saving both lines *and* tool results to each agent’s memory.

---

## 5. Extension ideas
* **Custom memory pruning** – subclass `SimpleMemory` to keep the last *k* tokens instead of *n* messages.  
* **Emotion tracking** – add an `emotion` attribute and tweak `generate_plan` / `generate_speech` prompts.  
* **Action validation** – override `generate_action` to auto‑parse Gemini’s output and retry on invalid JSON.

Pair these with a tweaked `World` or additional tools for richer simulations.

Happy agent‑crafting! 🤖