from agent import SimpleAgent, SimpleMemory, SelfCompressingMemory
from model import SimpleModel, GeminiModel, HTTPChatModel, RateLimitTracker, HFTransformersModel
import time

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


# freemodel = HFTransformersModel( "allenai/OLMo-2-0425-1B-Instruct", native_tool=False, )


# agent_mem = SimpleMemory()
# agent_mem = SelfCompressingMemory(100000,freemodel)
agent_mem = SelfCompressingMemory(10000,freemodel)

# Initially, copied from agent.py 
# Eventually: modify by hand or have LLM explore prompts agentically
default_plan_instruct_template = \
"""
First, identify what {self.name} would do.  Then make a very short plan to achieve those goals.  
Find a SMALL NUMBER of concrete steps that can be taken.  
Take available tools into account in your planning, but DO NOT do any tool calls.
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
    "investigator",
    plan_instruction_template = default_plan_instruct_template,
    action_instruction_template = default_action_instruct_template,
    speech_instruction_template = default_speech_instruct_template,
    memory = agent_mem,
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

import requests
def visit_url( url : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("Getting contents of =", url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
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
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        response_text = response.text
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
        summary_text = freemodel.response_text(summary)
        print("######### SUMMARY:\n",summary_text)
        return summary_text
    except Exception as e: 
        print(f"ERROR getting {url}, error {e}")
        return f"Error getting {url}"
visit_summary_tool = freemodel.register_tool(
    get_url_summary,
    "Visits the URL given by the user, and returns a summary of its contents.",
    {
        "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
    })

# def get_url_summary( url : str , **kwargs ):
#     try:
#         tracker_no_dos.wait()
#         print("Retrieving summarized url: ", url)

#         # Visit website and get its full contents
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()

#         response_text = response.text
#         # print("######### RESPONSE:\n",response_text)
        
#         # Extract information about the agent performing the search
#         agent : SimpleAgent = kwargs["agent"]
#         memory = agent.get_attribute("memory")
#         memory_str = str(memory)
#         # print("######### MEMORY:\n",memory_str)

#         # Use the agent's LLM to summarize the content
#         summary = agent.generate_content(
#             # instruction = "Summarize the INFORMATION in the website below.  Include all information relevant based on the memory.",
#             persona = "You are a research assistant. You process the raw HTML data of a website to the INFORMATION contained in the page.",
#             instruction = "Provide a DETAILED summary of all information this page relevant to your PLAN in YAML format.  See your MEMORY for detailed information about your plan and task.",
#             # instruction = "Write a list of all information this page in YAML format.  See your MEMORY for detailed information about your task.",
#             website = response_text,
#             memory = memory_str,
#         )
#         summary_text = freemodel.response_text(summary)
#         # print("######### SUMMARY:\n",summary_text)
#         return summary_text
#     except Exception as e: 
#         print(f"ERROR getting {url}, error {e}")
#         return f"Error getting {url}"
# visit_summary_tool = freemodel.register_tool(
#     get_url_summary,
#     "Visits the URL given by the user, and returns a summary of its contents.",
#     {
#         "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
#     })
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
mmqtest =  multimodal_query(
    "https://hips.hearstapps.com/hmg-prod/images/wisteria-in-bloom-royalty-free-image-1653423554.jpg", 
    "What is the color of this flower?", model=freemodel)
print("Multimodal query\n",mmqtest)

# mmqtest =  multimodal_query(
#     # "cca530fc-4052-43b2-b130-b30968d8aa44.png",
#     "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3",
#     "Describe the contents of this file",
#     #
#     # "https://hips.hearstapps.com/hmg-prod/images/wisteria-in-bloom-royalty-free-image-1653423554.jpg", 
#     # "What is the color of this flower?", 
#     model=freemodel)
# print("Multimodal query\n",mmqtest)

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
# tools = [visit_bleach_tool,search_tool,final_answer_tool]

agent.add_memory("My goal is to find the breed of the cat with the softest fur. \n I must complete this search in a limited number of stages.")

count = 0
# max_count = 1
max_count = 4
# max_count = 20
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
        print(freemodel.response_text(act))

    # (4) Use model.apply_tool.  Remember to supply all "hidden" arguments, not advertised to the model
    observe = freemodel.apply_tool(act, agent=agent)
    for result in observe:
        agent.add_memory("Observation: " + str(result[0]))

print(answer_string)