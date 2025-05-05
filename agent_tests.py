from agent import SimpleAgent, SimpleMemory
from model import GeminiModel

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

cust = SimpleAgent("Customer",
                   persona = "impatient, tech‑savvy",
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

def refund_purchase():
    print("Purchase refunded")
    exit
refund = freemodel.register_tool(refund_purchase,"The agent refunds the customer's purchase price.",{})    
def accept_resolution():
    print("Resolution accepted")
    exit
accept = freemodel.register_tool(accept_resolution,"The customer accepts the resolution.  Either the problem is fixed, or they have recieved a refund.",{})
import random
def check_phone():
    phone_status = random.choices(["Phone is now working","Phone is still broken"],[0.5,0.5])
    return phone_status
check = freemodel.register_tool(check_phone,"The customer checks the phone to see if it is working.  Outputs True or False",{})

observation = 'Customer says: "My jPhone-72 phone keeps rebooting!"'
mem_customer.add_memory(observation)
mem_support.add_memory(observation)
print(observation,"\n")
for _ in range(3) :
    # cont, sup_line = sup.respond(observation)
    # sup_line = sup.generate_speech(observation = observation )
    # sup_line = sup.generate_content(observation = observation , 
    #                                 # format = "Output a single sentence or paragraph."
    #                                 )
    sup_line = sup.generate_speech(observation = observation , 
                                    # format = "Output a single sentence or paragraph."
                                    )
    # sup_line = sup_line.text
    observation = "Support says: " + sup_line
    mem_customer.add_memory(observation)
    mem_support.add_memory(observation)
    print(observation,"\n")
    sup_action = sup.generate_action(model_parameters={"tools":[refund]})
    sup_action = freemodel.apply_tool(sup_action)
    for action in sup_action:
        description = "Support action: " + action[0][0]
        mem_customer.add_memory(description)
        mem_support.add_memory( description)
    print(sup_action)
    # result = "Support acts: " + sup_action
    # mem_customer.add_memory(result)
    # mem_support.add_memory(result)
    # print(result,"\n")
    # print(sup_line)
    # show("Support", sup_line)
    # print("")
    ## Don't trust the customer servie agent to end the conversation!
    # if not cont: break
    # cont, cust_line = cust.respond(sup_line)
    # cust_line = cust.generate_speech(observation = sup_line)
    # cust_line = cust.generate_content(observation = sup_line , 
    #                                 # format = "Output a single sentence or paragraph."
    #                                 )
    cust_line = cust.generate_speech(observation = sup_line , 
                                    # format = "Output a single sentence or paragraph."
                                    )
    # cust_line = cust_line.text
    observation = "Customer says: " + cust_line
    mem_customer.add_memory(observation)
    mem_support.add_memory(observation)
    print(observation,"\n")
    cust_action = cust.generate_action(model_parameters={"tools":[accept,check]})
    cust_action = freemodel.apply_tool(cust_action)
    print(cust_action)
    for action in cust_action:
        print("Desc", action[0][0])
        description = "Customer action:" + action[0][0]
        mem_customer.add_memory( description )
        mem_support.add_memory(  description )
        if action[0][0] == "Phone is now working": 
            exit

    print(cust_action)

    # result = "Customer acts: " + cust_action
    # mem_customer.add_memory(result)
    # mem_support.add_memory(result)
    # print(result,"\n")
    # show("Customer", cust_line)
    # if not cont: break
    observation = cust_line
