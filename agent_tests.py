from agent import SimpleAgent, SimpleMemory
from model import GeminiModel, HTTPChatModel, HFTransformersModel

"""## Run a (simple) conversation between agents."""

import pprint

customer_memories = [ "I bought a phone six weeks ago",
                      "My phone is an Android",
                      "I am hungry",
                      "I dropped my phone on cement",
                      "I brought my phone to the beach",
                      "I like to play chess"]
mem_customer = SimpleMemory( customer_memories )

mem_support  = SimpleMemory()

print(str(mem_support))

freemodel = GeminiModel("gemini-2.0-flash-lite", 30, 1000)
# freemodel = GeminiModel("gemini-2.5-flash-preview-04-17", 10, 1000)
# freemodel = HTTPChatModel("mistral-small:24b-instruct-2501-q4_K_M")
# freemodel = HTTPChatModel("gemma3:12b", 
#                           native_tool=False, 
#                         #   verbose=True,
#                           verbose=False,
#                           )


# freemodel = HFTransformersModel(
#   # "microsoft/Phi-4-mini-instruct",
#   "allenai/OLMo-2-0425-1B-Instruct",
#   native_tool=False,
# )


cust = SimpleAgent("Customer",
                   persona = "impatient, tech-savvy",
                   status = "has a phone that keeps restarting",
                   workflow = "Explain the issue.  Attempt to follow the instructions.  ALWAYS use `check` tool before saying whether the phone is still broken.  Only accept the resolution if the phone works or you get a refund.",
                   memory = mem_customer,
                   model = freemodel)
sup  = SimpleAgent("SupportAgent",
                   persona = "calm, helpful",
                   status = "waiting to assist",
                   workflow = "try very hard to debug the customer issue.  do not issue a refund unless (1) the phone is in the warantee period and (2) the problem is a result of a manufacturing defect.",
                   memory = mem_support,
                   model = freemodel)

## TOOLS 
def refund_purchase():
    sup.add_memory("I have refunded the customer's purchase.")
    cust.add_memory("My purchase has been refunded by support.")
    purchase_refunded = True
    print("Purchase refunded")
refund = freemodel.register_tool(refund_purchase,"The agent refunds the customer's purchase price.",{})    

keep_going = True
def accept_resolution():
    global keep_going
    keep_going = False
    print("Resolution accepted")
accept = freemodel.register_tool(accept_resolution,"The customer accepts the resolution.  Either the problem is fixed, or they have recieved a refund.",{})

import random
def check_phone():
    phone_status = random.choices(["Phone is now working","Phone is still broken"],[0.5,0.5])
    return phone_status
check = freemodel.register_tool(check_phone,"The customer checks the phone to see if it is working.  Outputs True or False",{})

## Start the conversation
observation = 'Customer says: "My jPhone-72 phone keeps rebooting!"'
mem_customer.add_memory(observation)
mem_support.add_memory(observation)
print(observation,"\n")
for _ in range(3) :
# for _ in range(30) :
    if not keep_going: 
        print("Conversation complete.")
        break

    # Support Speaks
    sup_line = sup.generate_speech(observation = observation , 
                                    # format = "Output a single sentence or paragraph."
                                    )
    sup_line = freemodel.response_text(sup_line)
    observation = "Support says: " + sup_line
    mem_customer.add_memory(observation)
    mem_support.add_memory(observation)
    print(observation,"\n")

    # Support Acts
    sup_action = sup.generate_action(model_parameters={"tools":[refund]},
                                     details="Respond 'No action taken' if you do not want to use any tool at this step.")
    sup_action = freemodel.apply_tool(sup_action)
    for action in sup_action:
        description = "Support action: " + str(action)
        mem_customer.add_memory(description)
        mem_support.add_memory( description)
        print(description)

    # Customer Speaks
    cust_line = cust.generate_speech(observation = sup_line , 
                                    # format = "Output a single sentence or paragraph."
                                    )
    print(cust_line)
    cust_line = freemodel.response_text(cust_line)
    observation = "Customer says: " + cust_line
    mem_customer.add_memory(observation)
    mem_support.add_memory(observation)
    print(observation,"\n")

    # Customer Acts
    cust_action = cust.generate_action(model_parameters={"tools":[accept,check]})
    cust_action = freemodel.apply_tool(cust_action)
    for action in cust_action:
        description = "Customer action:" + str(action)
        mem_customer.add_memory( description )
        mem_support.add_memory(  description )
        print(description)

    observation = cust_line
