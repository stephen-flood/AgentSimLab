from model import SimpleModel, RateLimitTracker
from agent import SimpleAgent
from googlesearch import search  # REQUIRES  pip install googlesearch-python 
from playwright.sync_api import sync_playwright
import requests
import bleach
# python -m pip install playwright
# python -m playwright install chromium


tracker_no_dos = RateLimitTracker(10)

def web_search( search_query : str , **kwargs ):
    try:
        tracker_no_dos.wait()
        print("searching")
        searchlist = search(search_query, num_results = 10, lang="en") 
        return list(searchlist)
    except:
        print("Error searching:",search_query)
        return ["Error in search for ",search_query]

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

        browser.close()

        # Return FULL HTML ( too much :( )
        # html = page.content()
        # return html

        try:
            # Return all plain text (no formatting/structure)
            text = page.evaluate("() => document.querySelector('article').innerText")
        except:
            # return FULL HTML of page
            text = page.content() 

        return text

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

def get_bleached_contents( url:str, **kwargs):
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

