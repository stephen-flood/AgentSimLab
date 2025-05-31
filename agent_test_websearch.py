from agent import SimpleAgent, SimpleMemory, SelfCompressingMemory
from model import GeminiModel

# freemodel = GeminiModel("gemini-2.0-flash-lite", 25, 1000)
# freemodel= GeminiModel("gemini-2.0-flash", 15, 1000)
# freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
freemodel= GeminiModel("gemini-2.5-flash-preview-04-17", 8, 1000)

# agent_mem = SimpleMemory()
agent_mem = SelfCompressingMemory(100000,freemodel)

agent = SimpleAgent(
    "investigator",
    memory = agent_mem,
    model = freemodel,
)



from googlesearch import search  # REQUIRES  pip install googlesearch-python 
def web_search( search_query : str , **kwargs ):
    try:
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
print(web_search("Cats"))

import requests
def visit_url( url : str , **kwargs ):
    try:
        response = requests.get(url, timeout=10)
        return response.text
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
    # try:
    # Visit website and get its full contents
    response = requests.get(url, timeout=10)
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
        instruction = "Write a list of all information this page in YAML format.  See your MEMORY for detailed information about your task.",
        website = response_text,
        memory = memory_str,
    )
    print("######### SUMMARY:\n",summary.text)
    return summary.text
    # except:
    #     print(f"ERROR getting {url}")
    #     return f"Error getting {url}"
visit_summary_tool = freemodel.register_tool(
    get_url_summary,
    "Visits the URL given by the user, and returns a summary of its contents.",
    {
        "url": {"type" : "string", "description" : "The URL of a website that I need to get the contents of."},
    })

# print(get_url_summary("https://en.wikipedia.org/wiki/Cat",agent=agent))



global finished
finished = False
answer_string = ""
def final_answer(answer : str, **kwargs):
    finished = True
    answer_string = answer
    print("Final answer:\n", answer_string)
final_answer_tool = freemodel.register_tool(
    func=final_answer,
    description="Use this function to report your final answer, and exit the search process.",
    parameters={
        "answer": {"type" : "string", "description" : "Your FINAL ANSWER to the original question."},
    }
)    
# tools = [visit_tool,search_tool,final_answer_tool]
tools = [visit_summary_tool,search_tool,final_answer_tool]

agent.add_memory("My goal is to find the breed of the cat with the softest fur. \n I must complete this search in a limited number of stages.")

count = 0
# max_count = 4
# max_count = 10
max_count = 5
while not finished and count < max_count:
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
    if act.function_calls is not None:
        print(act.function_calls)
        for call in act.function_calls:
            intended_action = f"Calling {call.name} with arguments {call.args}"
            agent.add_memory("Attempting Action: " + intended_action)
    else:
        print(act.text)

    # (4) Use model.apply_tool.  Remember to supply all "hidden" arguments, not advertised to the model
    observe = freemodel.apply_tool(act, agent=agent)
    for result in observe:
        agent.add_memory("Observation: " + str(result[0]))

print(answer_string)