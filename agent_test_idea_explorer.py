# pip install google-genai googlesearch-python bleach playwright
from agent import SimpleAgent, SimpleMemory, SelfCompressingMemory
from model import SimpleModel, GeminiModel, HTTPChatModel, RateLimitTracker, HFTransformersModel
import time

# Google free, NO native function calling
# freemodel = GeminiModel("gemma-3-27b-it", 25, 1000, native_tool = False)
#
# freemodel = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
freemodel= GeminiModel("gemini-2.0-flash", 15, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 8, 1000)

# cmd: ollama run <model_name>
# - mistral-small:24b-instruct-2501-q4_K_M
# - gemma3:27b
# - gemma3:12b
# freemodel = HTTPChatModel("gemma3:24b", native_tool=False)
# freemodel = HTTPChatModel("gemma3:12b", native_tool=False)
# freemodel = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")


# freemodel = HFTransformersModel( "allenai/OLMo-2-0425-1B-Instruct", native_tool=False, )


# agent_mem = SimpleMemory()
# agent_mem = SelfCompressingMemory(100000,freemodel)
agent_mem = SelfCompressingMemory(10000,freemodel)
plan_mem = SelfCompressingMemory(10000,freemodel)

# Initially, copied from agent.py 
# Eventually: modify by hand or have LLM explore prompts agentically
default_plan_instruct_template = \
"""
First, identify what {self.name} would do.  Then make a very short plan to achieve those goals.  
Find a SMALL NUMBER of concrete steps that can be taken.  
Take available tools into account in your planning, but DO NOT do any tool calls.
After the first stage, you should also DESCRIBE WHAT YOU LEARNED from previous observations.  THINK STEP BY STEP. 
"""
###
default_action_instruct_template = \
"""
What would {self.name} do? 
"""
###
default_speech_instruct_template = \
"""
What would {self.name} say?
"""

# Initialize agent
agent = SimpleAgent(
    "Explorer",
    plan_instruction_template = default_plan_instruct_template,
    action_instruction_template = default_action_instruct_template,
    speech_instruction_template = default_speech_instruct_template,
    #
    plan_memory = plan_mem,
    action_memory = agent_mem,
    #
    model = freemodel,
)

tracker_no_dos = RateLimitTracker(10)

from googlesearch import search  # REQUIRES  pip install googlesearch-python 
def web_search( search_query : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("searching")
        searchlist = search(search_query, num_results = 10, lang="en") 
        return list(searchlist)
    except:
        print("Error searching:",search_query)
        return ["Error in search for ",search_query]
# Step 2: create the tool object
search_tool = freemodel.register_tool(
    web_search,
    "Retrieves a list of website URL's relevant to the search query.",
    {
        "search_query": {"type" : "string", "description" : "A query for an online search.  This could be a question you want answered, a text fragment you want context for, the name of a file you are trying to find, or anything else."},
    })
# Quick test
# print("Testing Web Search.\n Search: 'cats'.  Response: ",web_search("Cats"))


## Use Chromium web browser and playwright to interpret JavaScript before parsing 
## Additional option: return only internal text of nodes
from playwright.sync_api import sync_playwright
# python -m pip install playwright
# python -m playwright install chromium
def fetch_html(url : str): 
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/125.0.0.0 Safari/537.36"),
            locale="en-US,en;q=0.9",
            viewport={'width': 1366, 'height': 768},
        )
        # page = browser.new_page()
        page = context.new_page()

        # Get page contents 
        response = page.goto(url ,
                #   wait_until="networkidle",
                # timeout : int =10000,
                )
        
        if not response:
            return f"ERROR accessing {url}:"
        elif not response.ok:
            return f"ERROR accessing {url}: {response.status}"

        # browser.close()

        # Return FULL HTML ( too much :( )
        # html = page.content()
        # return html

        # Return all plain text
        text = page.evaluate("() => document.querySelector('article').innerText")
        return text

# print("URL CONTENTS\n", fetch_html("https://medium.com/@Shamimw/i-struggled-with-pydantic-parser-because-i-chose-the-wrong-model-36fb11c6ec22"))
# print("URL CONTENTS\n", fetch_html("https://webwork.bridgew.edu/oer/Business_Calculus/ch-functions.html"))

import requests
def visit_url( url : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("Getting contents of =", url)

        # response = requests.get(url, timeout=10)
        # response.raise_for_status()
        # text = response.text

        text = fetch_html(url)

        return text
    except Exception as e:
        return f"Error getting {url}, with error {e}"
visit_tool = freemodel.register_tool(
    visit_url,
    "Visits the URL given by the user, and returns its full contents.",
    {
        "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
    })
# Quick test
visit_test_url = "https://www.charlottenc.gov/CATS/Home"
# print("URL CONTENTS\n", visit_url(visit_test_url))



import requests
def get_url_summary( url : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("Getting contents of =", url)

        # Visit website and get its full contents
        # response = requests.get(url, timeout=10)
        # response.raise_for_status()
        # response_text = response.text

        response_text = fetch_html(url)

        # print("######### RESPONSE:\n",response_text)
        
        # Extract information about the agent performing the search
        agent : SimpleAgent = kwargs["agent"]
        memory = agent.get_attribute("memory")
        memory_str = str(memory)
        # print("######### MEMORY:\n",memory_str)

        # Use the agent's LLM to summarize the content
        summary = agent.generate_content(
            # instruction = "Summarize the INFORMATION in the website below.  Include all information relevant based on the memory.",
            persona = "You are a research assistant. You summarize the  human readable information contained in a webpage.",
            instruction = "Provide a DETAILED description of the HUMAN READABLE CONTENT this page relevant to your PLAN in YAML format.  " \
            "See your MEMORY for detailed information about your plan and task.  " ,
            # instruction = "Write a list of all information this page in YAML format.  See your MEMORY for detailed information about your task.",
            website = response_text,
            memory = memory_str,
        )
        summary_text = freemodel.response_text(summary)
        print("######### SUMMARY:\n",summary_text)
        return summary_text
    except Exception as e: 
        print(f"ERROR getting {url}, error {e}")
        return f"Error getting {url}"
visit_summary_tool = freemodel.register_tool(
    get_url_summary,
    "Visits the URL given by the user, and returns a language model generated description of its raw contents.",
    {
        "url": {"type" : "string", "description" : "The URL of the website whose contents I am extracting."},
    })

## Easy test
# visit_test_url = "https://webwork.bridgew.edu/oer/Business_Calculus/ch-functions.html"
# Webpage with lots of formatting text (used for debugging INSTRUCTIONS) = REFUSES headless bot :(
# visit_test_url = "https://medium.com/@Shamimw/i-struggled-with-pydantic-parser-because-i-chose-the-wrong-model-36fb11c6ec22"
# 
# print("URL SUMMARY\n", get_url_summary(visit_test_url, agent=agent))

import bleach
def get_url_contents( url:str, **kwargs):
    try:
        tracker_no_dos.wait()

        print("Getting contents of =", url)
        # Visit website and get its full contents
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_text = response.text
        # response_text = fetch_html(url)
        # print("######### RESPONSE:\n",response_text)
        clean_text = bleach.clean(response_text,strip=True,strip_comments=True)
        clean_text = ' '.join(clean_text.split())
        return clean_text
    except Exception as e:
        print(f"ERROR getting {url}, error {e}")
        return f"Error getting {url}"
# print(get_url_summary("https://en.wikipedia.org/wiki/Cat",agent=agent))
visit_bleach_tool = freemodel.register_tool(
    get_url_contents,
    "Visits the URL given by the user, and returns a summary of its contents.",
    {
        "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
    })
# print("URL BLEACHED\n", get_url_contents(visit_test_url))


def multimodal_query( location: str, query: str, *, model : SimpleModel):
    try:
        response = model.generate_content(
                user_prompt      = query,
                attachment_names =[location],
            )
        print(response)
        return freemodel.response_text(response)
    except Exception as e:
        print("Error with Multimodal Prompt:", e)
        return "Error with Multimodal Prompt:" + str(e)

multimodal_query_tool = freemodel.register_tool(
    multimodal_query,
    "Answers questions about multimodal inputs such as images and videos.",
    {
        "query":    {"type" : "string", "description" : "The question I want to ask about the image or video."},
        "location": {"type" : "string", "description" : "The filename or url of the image or video I am interested in."},
    })
# mmqtest =  multimodal_query(
#     "https://hips.hearstapps.com/hmg-prod/images/wisteria-in-bloom-royalty-free-image-1653423554.jpg", 
#     "What is the color of this flower?", model=freemodel)
# print("Multimodal query\n",mmqtest)


keep_going  = True
answer_string = ""
def report_final_answer(answer : str, **kwargs):
    global keep_going , answer_string
    keep_going = False
    answer_string = answer
    print("REPORTIN FINAL ANSWER:\n", answer_string)
final_answer_tool = freemodel.register_tool(
    func=report_final_answer,
    description="Use this function to report your final answer, and exit the search process.",
    parameters={
        "answer": {"type" : "string", "description" : "Your FINAL ANSWER to the original question."},
    }
)    
tools = [visit_tool,search_tool,final_answer_tool]
# tools = [visit_summary_tool,search_tool,final_answer_tool]  # Better when context window is SMALLER
# tools = [visit_bleach_tool,search_tool,final_answer_tool] # Better for LLMs with BIG context window

import yaml
default_log_name = "agent_test_idea_explorer_history.txt"
def log_information( category: str , information : str, log_name : str = default_log_name):
    try:
        with open( log_name, "r") as file:
            log = yaml.safe_load(file)
    except:
        log = []

    log.append( { category: information } )

    with open(log_name, "w") as file:
        yaml.dump( log , file)

# topic = "Agents and MCP (Model Context Protocol) servers."
# topic = "What is involved in setting up a MCP (Model Context Protocol) server?  What are some problems that it can solve?  " \
#     "What steps are required to set one up?  Include sample code for a MIMIAL Hello World MCP."
# topic = "I want to create my own VERY SIMPLE implementation of a coding agent (such as windsurf, codex, claude code).  " \
#     "I want the agent to be running inside its own docker image, and have access to its own forked copy of a github repository. " \
#     "I already have an agent library which handles the agent's internal memories, generation of plans/actions, and handles tool calls. " \
#     "Can you give me a minimial, very simple implementation in python?  "
topic = "It seems like there is some drama between the people at pydantic and langchain.  " \
    "Can you explain whats going on here?  Is this all just smoke, or is there some important substance here?  " \
    "In your final report, give me SPECIFIC LINKS to some of the research you came up with to explain your conclusions."
agent.set_attribute(
    key = "Objective",
    value =\
f"""
I am an avid learner.  
My goal is to learn as much as I can about the following TOPIC.  
I should prioritize authoritative sources like blogposts by major AI labs, 
    and companies developing libraries for AI use.  
TOPIC: {topic}
APPROACH: I need to 
1. Identify a new important topic.
2. Perform in depth research into this topic. 
3. Write a DETAILED final report (1 page) describing my discoveries. INCLUDE the report as the input `answer=` to the function `report_final_answer` .
"""
)

log_information("Starting Exploration", agent.description())
log_information("Model type", freemodel.model_name )
log_information("Tools", str(tools))

    #  = plan_mem,
    #  = agent_mem,

count = 0
# max_count = 1
# max_count = 4
# max_count = 10
# max_count = 20
max_count = 30
# max_count = 5
while keep_going and (count < max_count):
    count += 1

    print(f"########### Cycle {count} of {max_count} ###########")

    plan = agent.generate_plan( 
        details=f"I am on agent step {count}/{max_count}.  If this is the last step, I MUST call the final answer tool with my final answer.  \
                  I CAN call the final answer tool early, as soon as I have completed my research, but NOT until I have used MULTIPLE rounds \
                  of tool calling to investigate and document my report." ,  
        # tools=tools ## Avoid tools in planning phase - o/w model just tries to call a tool!
    )
    # agent.add_memory("Plan: " + str(plan))
    agent.add_memory("Plan: " + str(plan), memory_variable="plan_memory")
    print("PLAN\n", plan)
    log_information("Plan" , str(plan) )

    # (3) Supply tool objects during agent.generate_action
    act = agent.generate_action(
        ## Better models
        # system="Identify a TOOL that will help achieve your plan, and return a TOOL CALL",
        ## Weaker models
        system="You are a tool use model.  You MUST identify a TOOL that will help achieve your plan, and return a TOOL CALL.  You should ALTERNATE web search and visiting websites.  If access is denied, try a DIFFERENT website next time.",
        plan=plan,
        tools=tools,
    )
    act_calls = freemodel._iter_tool_calls(act) 
    if act_calls is not None:
        # print(act_calls)
        for call in act_calls:
            # print(call)
            intended_action = f"Calling {call[0]} with arguments {call[1]}"
            # agent.add_memory("Attempting Action: " + intended_action)
            agent.add_memory("Attempting Action: " + intended_action, memory_variable="action_memory")
            log_information("Attempting Action", str(intended_action) )
            print("ATTEMPTING ACTION:\n", str(intended_action))
    # if act.function_calls is not None:
    #     print(act.function_calls)
    #     for call in act.function_calls:
    #         intended_action = f"Calling {call.name} with arguments {call.args}"
    #         agent.add_memory("Attempting Action: " + intended_action)
    else:
        print(freemodel.response_text(act))
        log_information("No Action.  LLM Text", freemodel.response_text(act))

    # (4) Use model.apply_tool.  Remember to supply all "hidden" arguments, not advertised to the model
    observe = freemodel.apply_tool(act, agent=agent)
    for result in observe:
        agent.add_memory("Observation: " + str(result[0]), memory_variable="action_memory")
        # agent.add_memory("Observation: " + str(result[0]))
        log_information("Observation" , str(result[0]))
        print("OBSERVATION:\n", str(result[0]))

print(answer_string)
log_information( "Final Answer" , answer_string )