from agent import SimpleAgent, SimpleMemory, show
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

freemodel = GeminiModel("gemini-2.0-flash-lite", 30, 1000)

cust = SimpleAgent("Customer",
                   "impatient, tech‑savvy",
                   "has a phone that keeps restarting",
                   mem_customer,
                   freemodel)
sup  = SimpleAgent("SupportAgent",
                   "calm, helpful",
                   "waiting to assist",
                   mem_support,
                   freemodel)

observation = 'Customer says: "My phone keeps rebooting!"'
print(observation,"\n")
for _ in range(30):
    cont, sup_line = sup.respond(observation)
    # print(sup_line)
    show(*sup_line)
    print("")
    ## Don't trust the customer servie agent to end the conversation!
    # if not cont: break
    cont, cust_line = cust.respond(sup_line)
    show(*cust_line)
    print("")
    # if not cont: break
    observation = cust_line
