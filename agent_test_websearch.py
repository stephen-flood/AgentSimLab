from agent import SimpleAgent, SimpleMemory, SelfCompressingMemory
from model import GeminiModel, HTTPChatModel, RateLimitTracker
import time

# Google free, NO native function calling
# freemodel = GeminiModel("gemma-3-27b-it", 25, 1000)
#
# freemodel = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
# freemodel= GeminiModel("gemini-2.0-flash", 15, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 8, 1000)

# cmd: ollama run <model_name>
# - mistral-small:24b-instruct-2501-q4_K_M
# - gemma3:27b
# - gemma3:12b
# freemodel = HTTPChatModel("gemma3:24b", native_tool=False)
freemodel = HTTPChatModel("gemma3:12b", native_tool=False)
# freemodel = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")

# agent_mem = SimpleMemory()
agent_mem = SelfCompressingMemory(100000,freemodel)

agent = SimpleAgent(
    "investigator",
    memory = agent_mem,
    model = freemodel,
)

tracker_no_dos = RateLimitTracker(10)

from googlesearch import search  # REQUIRES  pip install googlesearch-python 
def web_search( search_query : str , **kwargs ):
    try:
        tracker_no_dos.log_query()
        wait = tracker_no_dos.time_to_wait()
        if wait:
            time.sleep(wait)
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
print("Testing Web Search.\n Search: 'cats'.  Response: ",web_search("Cats"))

import requests
def visit_url( url : str , **kwargs ):
    try:
        tracker_no_dos.log_query()
        wait = tracker_no_dos.time_to_wait()
        if wait:
            time.sleep(wait)
        response = requests.get(url, timeout=10)
        # return response.text
        return freemodel._response_text(response)
    except:
        return f"Error getting {url}"
visit_tool = freemodel.register_tool(
    visit_url,
    "Visits the URL given by the user, and returns its full contents.",
    {
        "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
    })
# Quick test
# print(visit_url("http://www.google.com"))
# print(visit_url(str(web_search("Cats")[0])))

import requests
def get_url_summary( url : str , **kwargs ):
    try:
        # Visit website and get its full contents
        response = requests.get(url, timeout=10)
        # response_text = response.text
        response_text = freemodel._response_text(response)
        # print("######### RESPONSE:\n",response_text)
        
        # Extract information about the agent performing the search
        agent : SimpleAgent = kwargs["agent"]
        memory = agent.get_attribute("memory")
        memory_str = str(memory)
        # print("######### MEMORY:\n",memory_str)

        # Use the agent's LLM to summarize the content
        summary = agent.generate_content(
            # instruction = "Summarize the INFORMATION in the website below.  Include all information relevant based on the memory.",
            persona = "You are a research assistant. You process the raw HTML data of a website to the INFORMATION contained in the page.",
            instruction = "Provide a DETAILED summary of all information this page relevant to your PLAN in YAML format.  See your MEMORY for detailed information about your plan and task.",
            # instruction = "Write a list of all information this page in YAML format.  See your MEMORY for detailed information about your task.",
            website = response_text,
            memory = memory_str,
        )
        summary_text = freemodel._response_text(summary)
        print("######### SUMMARY:\n",summary_text)
        return summary_text
    except:
        print(f"ERROR getting {url}")
        return f"Error getting {url}"
visit_summary_tool = freemodel.register_tool(
    get_url_summary,
    "Visits the URL given by the user, and returns a summary of its contents.",
    {
        "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
    })

# print(get_url_summary("https://en.wikipedia.org/wiki/Cat",agent=agent))



keep_going  = True
answer_string = ""
def report_final_answer(answer : str, **kwargs):
    global keep_going 
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
# tools = [visit_tool,search_tool,final_answer_tool]
tools = [visit_summary_tool,search_tool,final_answer_tool]

agent.add_memory("My goal is to find the breed of the cat with the softest fur. \n I must complete this search in a limited number of stages.")

count = 0
# max_count = 1
# max_count = 4
max_count = 20
# max_count = 5
while keep_going and (count < max_count):
    count += 1

    print(f"########### Cycle {count} of {max_count} ###########")

    plan = agent.generate_plan( details=f"I am on agent step {count}/{max_count}.  If this is the last step, I MUST call final_answer_tool with my final answer.  I CAN call final_answer_tool early, as soon as I have completed my research." ,  tools=tools)
    agent.add_memory("Plan: " + str(plan))
    print(plan)

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
            agent.add_memory("Attempting Action: " + intended_action)
    # if act.function_calls is not None:
    #     print(act.function_calls)
    #     for call in act.function_calls:
    #         intended_action = f"Calling {call.name} with arguments {call.args}"
    #         agent.add_memory("Attempting Action: " + intended_action)
    else:
        print(freemodel._response_text(act))

    # (4) Use model.apply_tool.  Remember to supply all "hidden" arguments, not advertised to the model
    observe = freemodel.apply_tool(act, agent=agent)
    for result in observe:
        agent.add_memory("Observation: " + str(result[0]))

print(answer_string)