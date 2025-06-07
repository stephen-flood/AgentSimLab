from model import GeminiModel, HTTPChatModel, RateLimitTracker

# Google free, NO native function calling
# freemodel = GeminiModel("gemma-3-27b-it", 25, 1000)
#
freemodel = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
# freemodel= GeminiModel("gemini-2.0-flash", 15, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 8, 1000)

# cmd: ollama run <model_name>
# - mistral-small:24b-instruct-2501-q4_K_M
# - gemma3:27b
# - gemma3:12b
# freemodel = HTTPChatModel("gemma3:24b", native_tool=False)
# freemodel = HTTPChatModel("gemma3:12b", native_tool=False)
# freemodel = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")

from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

PAGE = """
<!doctype html>
<head>
  <title>SimpleCodeAgent</title>
  <script id="MathJax-script"
        async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>

  <style> /* o3 */
    /* --- ONE tiny typographic reset ------------------------------------ */
    body { font-family: system-ui, sans-serif; margin: 1.5rem; line-height: 1.4; }

    /* --- Form bits ------------------------------------------------------ */
    textarea  { width: 100%; max-width: 100%; font-family: inherit; }
    button    { padding: .5rem 1rem; border: 0; border-radius: 6px;
                background:#1976d2; color: #fff; cursor: pointer; }

    /* --- “bubble” style ---------------------------------------- */
    .chat-item {
      border: 1px solid #dcdcdc;       /* subtle line */
      border-radius: 8px;              /* rounded corners */
      padding: .75rem 1rem;
      margin-bottom: .75rem;
      background: #fafafa;
      white-space: pre-wrap;           /* keep line-breaks from Markdown */
    }
  </style>
</head>
<body>
  <h1>Coding Agent</h1>

  <form method="post">
    <input name="user_input" type="text" placeholder="Type your prompt here" autofocus>
    <button type="submit">Submit</button>
  </form>

  {% if user_input %}
    <pre class="chat-item">User: {{ user_input }}</pre>
  {% endif %}
  {% if clean_output %}
    <div class="chat-item">Assistant: {{ clean_output | safe }}</div>
  {% endif %}

  <div class = "history">
    <h2> Chat History </h2>
  {% for element in history %}
    <div class="role"> 
        <h3>{{element["role"].title()}}</h3>
    </div>
    <!--pre class="chat-item">History: Text {{ element["content"] }}</pre-->
    <div class="chat-item">History: Text {{ element["content"] | safe }}</div>
  {% endfor %}
  </div>
</body>
</html>
"""
# Main page performs LLM queries and stores responses.
## Define behavior for access to /
history = []
@app.route("/", methods=["GET", "POST"])
def route_index():
    global history
    user_input = request.form.get("user_input") if request.method == "POST" else None
    # print("Input: ", user_input)
    response = freemodel.generate_content(user_prompt = user_input)
    # print("Response:" , response)
    response_text = freemodel.response_text(response)
    response_html = md_to_safe_html(response_text)
    history.append(
        {
            "role" : "user",
            "content" : user_input
        }
    )
    history.append(
        {
            "role" : "assistant",
            "content" : response_html
        }
    )
    return render_template_string(PAGE, 
                                  user_input = user_input,
                                  clean_output = response_html,
                                  history= history,
                                  )

# From o3:
import markdown, bleach

def md_to_safe_html(markdown_text: str) -> str:
    raw_html = markdown.markdown(
        markdown_text,
        extensions=["extra", "codehilite"]  # whatever features you need
    )
    # Strip anything dangerous like <script> (Bleach, default allow-list)
    clean_html = bleach.clean(raw_html, strip=True)
    return clean_html

# Test persistent storage using global variables.
counter_storage = 0
## Define behavior for access to /increment
@app.route("/increment", methods=["GET","POST"])
def _():
    global counter_storage
    n = int(request.values.get("n", 1))
    counter_storage += n              # global state for demo
    return jsonify(value=counter_storage) 


# Run flask
if __name__ == "__main__":
    app.run( debug=True , host="0.0.0.0",  port=5000)
