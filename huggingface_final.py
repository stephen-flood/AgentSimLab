from agent import SimpleAgent, SimpleMemory, SelfCompressingMemory
from model import SimpleModel, GeminiModel, HTTPChatModel, RateLimitTracker, HFTransformersModel
from pathlib import Path
import time
import requests
import yaml


######## HUGGINGFACE COURSE HELPER FUNCTIONS ########
def get_all_questions():
  url = 'https://agents-course-unit4-scoring.hf.space/questions'
  headers = {'accept': 'application/json'}
  response = requests.get(url, headers=headers)
  response.raise_for_status()          
  list_of_problems = response.json()   
  return list_of_problems

def get_url_from_name(file_name):
  task_id = file_name.split(".")[0]
  return 'https://agents-course-unit4-scoring.hf.space/files/' + task_id

problem_file = "huggingface_final_questions.txt"
try:
  print("Opening question file")
  with open(problem_file,"r") as file:
    question_dictionary = yaml.safe_load(file)
except:
  print("Fetching questions")
  question_dictionary = get_all_questions()
  with open(problem_file,"w") as file:
    yaml.dump(question_dictionary,file)



######## LOAD MODEL WRAPPER from `model.py` ########
# - SimpleModel subclasses advertise same `generate_content` and tool related methods
# - Same code works for models that ARE and are NOT native-tool models
# - Allows quick comparison of different models and libraries

## Official google-genai models
# freemodel = GeminiModel("gemma-3-27b-it", 25, 1000)
# freemodel = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
# freemodel= GeminiModel("gemini-2.0-flash", 15, 1000)
freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 5, 1000)

## General access to OpenAPI models
## Here, used locally with `ollama run <model_name>`
# freemodel = HTTPChatModel("gemma3:24b", native_tool=False)
# freemodel = HTTPChatModel("gemma3:12b", native_tool=False)
# freemodel = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")

## HuggingFace Transformers
# freemodel = HFTransformersModel( "allenai/OLMo-2-0425-1B-Instruct", native_tool=False, )


######## INITIALIZE AGENT and MEMORY from `agent.py` ########

# - Each agent will have a single memory repository
# - Memory will be summarized by an LLM when it reaches a character limit
# - Default agent plan and action instructions overriden as specified
# - Agent also recieves an extra "format" attribute 
#   which will be appended into all of its prompts

# agent_mem = SimpleMemory()
# agent_mem = SelfCompressingMemory(10000,freemodel)
agent_mem = SelfCompressingMemory(50000,freemodel)
# agent_mem = SelfCompressingMemory(100000,freemodel)

default_plan_instruct_template = \
"""
First, identify what {self.name} would do.  Then make a very short plan to achieve those goals.  
Find a SMALL NUMBER of concrete steps that can be taken.  
Take available tools into account in your planning, but DO NOT do any tool calls.
After the first stage, you should also DESCRIBE WHAT YOU LEARNED from previous observations.  THINK STEP BY STEP. 
"""
default_action_instruct_template = \
"""
What would {self.name} do? 
"""

# Initialize agent object
agent = SimpleAgent(
    "Research Assistant",
    plan_instruction_template = \
"""
First, identify what {self.name} would do.  Then make a very short plan to achieve those goals.  
Find a SMALL NUMBER of concrete steps that can be taken.  
Think about tools into account in your planning.
Return a TEXT plan only.  
DO NOT RETURN any tool calls.
""",
    action_instruction_template = \
"""
You are {self.name}. 
A plan and past observations are stored in your MEMORY.
Consult your MEMORY and available TOOLS.   
If appropriate, perform and appropriate TOOL CALL.
Make sure you use the correct function and argument names. 
""",
    # Format (modified) from Gaia Benchmark (https://huggingface.co/spaces/gaia-benchmark/leaderboard)
    format = "You are a general AI assistant. I will ask you a question. " \
             "Report your thoughts at each step. " \
             "Use the `report_final_answer` tool to report your FINAL ANSWER. " \
             "Your FINAL ANSWER should be a number OR as few words as possible " \
                "OR a comma separated list of numbers and/or strings. " \
             "If you are asked for a number, don't use comma to write your number " \
                "neither use units such as $ or percent sign unless specified otherwise. " \
             "If you are asked for a string, don't use articles, neither abbreviations "
                "(e.g. for cities), and write the digits in plain text unless specified otherwise. " \
             "If you are asked for a comma separated list, " \
                "apply the above rules depending of whether the element to be put in the list "
                "is a number or a string.",
    memory = agent_mem,
    model = freemodel,
)



######## INITIALIZE TOOLS and REGISTER WITH MODEL ########
# - google search 
# - retrieve markdown of url contents
# - multimodal queries (can be used with separate SimpleModel for text only models)
# - report final answer


# Be a good web citizen - one websearch or retrieval per 6s
# just call `tracker_no_dos.wait()`
tracker_no_dos = RateLimitTracker(10)

#### TOOL 1: GOOGLE SEARCH ####
from googlesearch import search  
def web_search( search_query : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("Searching for:", search_query)
        searchlist = search(search_query, num_results = 10, lang="en") 
        return list(searchlist)
    except:
        print("Error searching:",search_query)
        return ["Error in search for ",search_query]
search_tool = freemodel.register_tool(
    web_search,
    "Retrieves a list of website URL's relevant to the search query.",
    {
        "search_query": {"type" : "string", "description" : "A query for an online search.  This could be a question you want answered, a text fragment you want context for, the name of a file you are trying to find, or anything else."},
    })

#### TOOL 2: RETRIEVE (cleaned up) URL CONTENTS #### 
from playwright.sync_api import sync_playwright
import html2text
def fetch_html(url : str): 
    """
    Use playwright to access contents of Javascript generated websites, 
    Optionally process page contents
        - full html converted to markdown
        - internal text only
        - other?
    """
    with sync_playwright() as p:
        # create browser, context, and page
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/125.0.0.0 Safari/537.36"),
            locale="en-US,en;q=0.9",
            viewport={'width': 1366, 'height': 768},
        )
        page = context.new_page()

        # Get page contents 
        response = page.goto(url)
        if not response:
            return f"ERROR accessing {url}:"
        elif not response.ok:
            return f"ERROR accessing {url}: {response.status}"
        # Process contents
        try:
            print("returning html as markdown")
            # return FULL HTML of page
            rawhtml = page.content() 
            text = html2text.html2text(rawhtml)
        except:
            print("returning contents")
            # Return all plain text (no formatting/structure)
            text = page.evaluate("() => document.body.innerText")
 
        browser.close()

        return text
def visit_url( url : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("Getting contents of =", url)
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



#### TOOL 3: Multimodal queries #### 

def multimodal_query( location: str, query: str, *, model : SimpleModel, **kwargs):
    try:
        print(f"Sending multimodal query: {query} \n about file at location: {location}")

        # For HuggingFace course
        # Use their API to get the files with 'local' names. 
        if "http" not in location:
            path = Path(location)

            if path.exists():
               print(f"Using existing file: {location}") 
            else:
                print(f"Saving new file: {location}")
                remote_location = get_url_from_name(location)
                file_contents = requests.get(remote_location).content
                with open(location,"wb") as f:
                    f.write(file_contents)
        # Process actual query
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



#### TOOL 4: Report Final Answer #### 

keep_going  = True
answer_string = ""
def report_final_answer(answer : str, **kwargs):
    global keep_going, answer_string
    keep_going = False
    answer_string = answer
    print("Final answer:\n", answer_string)
final_answer_tool = freemodel.register_tool(
    func=report_final_answer,
    description="Use this function to report your final answer, and exit the search process.",
    parameters={
        "answer": {"type" : "string", "description" : "Your FINAL ANSWER to the original question."},
    }
)    

#### Set list of tools to be used 
tools = [visit_tool,search_tool,multimodal_query_tool,final_answer_tool]



######## AGENT EXECUTION LOOP 

# question_subset = [question_dictionary[0]]
# question_subset = question_dictionary[4:5]
question_subset = question_dictionary
for stage, raw_task in enumerate( question_subset ):
    print(f"================ Question {stage+1} of {len(question_subset) } ================")
    task = {
        "question" : raw_task.get("question"),
        "file_name": raw_task.get("file_name")
    }
    agent.set_attribute(
        key = "Objective", 
        value = task
    )

    print(agent.description())

    count = 0
    # max_count = 1
    # max_count = 4
    max_count = 5
    # max_count = 200
    while keep_going and (count < max_count):
        count += 1

        print(f"########### Phase {count}/{max_count} in Question {stage+1}/{len(question_subset)} ###########")

        #### PHASE 1: PLAN
        plan = agent.generate_plan( details=f"I am on agent step {count}/{max_count}.  If this is the last step, I MUST call final_answer_tool with my final answer.  I CAN call final_answer_tool early, as soon as I have completed my research." ,  tools=tools)
        agent.add_memory("Plan: " + str(plan))
        print(plan)

        #### PHASE 2: DETERMINE ACTION
        act = agent.generate_action(
            ## Better models
            # system="Identify a TOOL that will help achieve your plan, and return a TOOL CALL",
            ## Weaker models
            system="You are a tool use model.  " \
            "You MUST identify a TOOL that will help achieve your plan, "
            "and return a TOOL CALL.  You should ALTERNATE web search and visiting websites.  " \
            "If access is denied, try a DIFFERENT website next time.",
            plan=plan,
            tools=tools,
        )

        #### PHASE 3a: RECORD INTENDED ACTION
        # ( Useful to model if it needs to fix tool syntax or handle broken website next time)
        act_calls = freemodel._iter_tool_calls(act) 
        if act_calls is not None:
            print("Function calls detected.")
            for call in act_calls:
                intended_action = f"Calling {call[0]} with arguments {call[1]}"
                agent.add_memory("Attempting Action: " + intended_action)
                print(intended_action)
        else:
            print("No tool action. Text:", freemodel.response_text(act))

        #### PHASE 3b: ATTEMPT INTENDED ACTION
        observe = freemodel.apply_tool(act, agent=agent, model=freemodel)
        for result in observe:
            agent.add_memory("Observation: " + str(result[0]))

    raw_task["submitted_answer"] = str(answer_string)
    print("Saving answer:", answer_string)

    # Reset the variables needed for a new investigation 
    agent.clear_memory()
    keep_going = True
    answer_string = ""


######## PRINT RECORD OF MODEL RESULTS TO DISK ########

model_output = {
    "model" : freemodel.model_name,
    "tools_used" : tools,
    "questions" : question_subset,
}

from datetime import datetime
now = datetime.now()
print(now)
time = now.strftime("%Y%m%d-%H%M")
model_name = freemodel.model_name
filename = "huggingface_final_output/" + model_name + time + ".txt"
with open(filename, "w") as file:
    print(f"Saving output to {filename}")
    yaml.dump(model_output, file)