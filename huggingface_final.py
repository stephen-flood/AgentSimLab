from model import GeminiModel, Prompt
from agent import SimpleAgent
import requests

def get_all_questions():
  url = 'https://agents-course-unit4-scoring.hf.space/questions'
  headers = {'accept': 'application/json'}
  response = requests.get(url, headers=headers)
  response.raise_for_status()         # throws if status is 4xx/5xx
  list_of_problems = response.json()   # directly gives you the parsed dict
  return list_of_problems

def get_url_from_name(file_name):
  return 'https://agents-course-unit4-scoring.hf.space/files/' + file_name

import pprint
question_dictionary = get_all_questions()
pprint.pp(question_dictionary)


## TODO: 1. Create TOOLS
#              a. "GOOGLE SEARCH" tool object 
#              b. "WGET" tool object (or perfectsoup tool?)
#              c. "FINAL_ANSWER" tool object (otherwise, keep the loop going)

## TODO: 2. Create AGENT with these tools, have them go through the ReAct cycle