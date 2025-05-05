# `model.py` – Deep Dive Reference
A detailed walkthrough of the helper classes that wrap Google Gemini for **rate-limited, multimodal, tool-augmented queries**.  Designed for students who want to read the source and extend it on their own.


---

## 1. Big-picture philosophy
* **Transparency › Magic** – every helper is a thin Python layer you can open, copy, or strip away.  
* **Hackability › Lock-in** – static methods avoided; most attributes are plain instance variables you can patch at runtime.  
* **Zero-friction prompts** – a `dict → YAML` one-liner via the `Prompt` class.  
* **Respect free-tier limits** – the `RateLimitTracker` stays out of the way unless you approach quota.

---

## 2. Environment & setup
<CODE>
# Inside model.py
try:
    from google.colab import userdata          # 👋 Colab users
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
except:
    with open("gemini.api_key") as fh:          # 👋 local / notebook users
        os.environ["GEMINI_API_KEY"] = fh.read()
</CODE>
If neither source exists you will hit an auth error when instantiating `GeminiModel` – make sure one of the two paths sets the `GEMINI_API_KEY` env var.

---

## 3. `RateLimitTracker`
<CODE>
tracker = RateLimitTracker(per_min_limit=30)
...
tracker.log_query()
wait = tracker.time_to_wait()
</CODE>
| Property | Meaning |
|----------|---------|
| `per_min_limit` | how many calls you *think* Google allows per minute |
| `request_history_minute` | sliding window of timestamps |

### How the wait time is chosen
1. First **¼** of queries: no delay – burst quickly ≤ 7 calls.  
2. Second **¼**: half-speed – short bursts still ok.  
3. Final **½**: slow lane – ensures you never exceed the cap.

Feel free to swap the policy with your own formula; the class is only ~25 lines.

---

## 4. `GeminiModel`
### 4.1 Constructor
<CODE>
llm = GeminiModel("gemini-2.0-flash-lite", rate_limit_minutes=30, rate_limit_daily=1000)
</CODE>
*Stores a `genai.Client`, a `RateLimitTracker`, and an in-memory `tool_registry`*.

### 4.2 `generate_content(**kwargs)` – one API to rule them all
Supported keys:  
* `user_prompt` – plain text → converted into a `types.Content` block.  
* `contents` – full manually-built list of `Content` / `Part` objects.  
* `tools` – list of tool objects (see 4.4).

Behind the scenes:  
1. Builds a `GenerateContentConfig` (adds `tools` if given).  
2. Calls `self.rate_limit_tracker` → sleeps if needed.  
3. Sends the request and returns the raw Gemini `GenerateContentResponse`.

### 4.3 `multimodal_query()` – quick helper
<CODE>
resp = llm.multimodal_query(
    user_prompt="What animal is shown here?",
    attachment_names=["https://example.com/penguin.jpg"]
)
print(resp.text)
</CODE>
* Fetches each URL / local path → `Part` with correct MIME type.  
* Optional `system_prompt` goes in as a `role="system"` block.

### 4.4 Tool Registry
<CODE>
def cube(x:int)->int: return x**3
cube_tool = llm.register_tool(
    func=cube,
    description="Returns the cube of an integer.",
    parameters={"x": {"type":"integer","description":"Number to cube"}}
)

resp = llm.generate_content(user_prompt="What is 7³?", tools=[cube_tool])
print(llm.apply_tool(resp))   # [('343', <FunctionCall …>)]
</CODE>
* Registration stores both the Python function and the Gemini tool object under the same name → no globals needed.

### 4.5 `apply_tool()` – run whatever Gemini called
Merges the model-supplied JSON args with any hard-coded kwargs you pass in, then invokes the matching Python function. Returns a list of `(result_text, call)` tuples so you can log or feed them back into an agent.

---

## 5. `Prompt` helper
### 5.1 Design goals
* Accept any keys – unknown ones get appended after the standard block.  
* Output readable YAML so humans can sanity-check the final prompt.  
* Repeat the instruction at the bottom – empirical trick to keep Gemini on-task.

### 5.2 Minimal example
<CODE>
from model import Prompt
spec = {
    "persona": "You are an adventurous marine biologist and gifted storyteller.",
    "context": "You are recording a segment for a children's science podcast about the ocean.",
    "instruction": "Explain why protecting coral reefs matters.",
    "tone": "inspiring and vivid"
}
print(Prompt(**spec).generate())
</CODE>
Output (excerpt):  
<CODE>
Persona: You are an adventurous marine biologist and gifted storyteller.
Context: You are recording a segment for a children's science podcast about the ocean.
Instruction: Explain why protecting coral reefs matters.
...
Remember: the instruction is Explain why protecting coral reefs matters.
</CODE>

---

## 6. End-to-end example (adapted from `model_tests.py`)
<CODE>
from model import GeminiModel, Prompt

llm = GeminiModel()

# 1) Build a prompt
prompt = Prompt(
    persona="exhausted pirate",
    context="A rival ship appears on the horizon",
    instruction="Utter a single dramatic warning",
)

# 2) Ask the model – no tools this time
resp = llm.generate_content(user_prompt=prompt.generate())
print(resp.text)
</CODE>
Try editing one field at a time and observe how the YAML dump & output change.

---

## 7. Extending the module
* Custom rate-limit strategy – subclass `RateLimitTracker` and slot it into your own `GeminiModel` variant.  
* Persistent tool registry – serialize `self.tool_registry` to JSON if you need hot-reloads in a long-running app.  
* Richer prompt templates – derive from `Prompt` and override `.generate()` to output Markdown, XML, or whatever your workflow prefers.

Happy experimenting! 🎉