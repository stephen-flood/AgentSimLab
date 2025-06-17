from model import SimpleModel, RateLimitTracker
from agent import SimpleAgent
from googlesearch import search  
    # pip install googlesearch-python 
from playwright.sync_api import sync_playwright 
    # python -m pip install playwright
    # python -m playwright install chromium
import requests
import bleach
import html2text


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

test_query = "What is the height of the tallest skyscraper?"
print("WEB SEARCH TEST\n", web_search(test_query))


def get_url( url : str ):
    """ Use requests library to get contents """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def robust_get_page_text(url : str): 
    """
    Use playwright to access contents of Javascript generated websites, 
    Optionally process page contents
        - full html converted to markdown
        - internal text only
        - other?
    """
    with sync_playwright() as p:
        # create browser, context, and page
        # provide a plausible context to avoid blockers
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
# test_url = "https://medium.com/@Shamimw/i-struggled-with-pydantic-parser-because-i-chose-the-wrong-model-36fb11c6ec22"
# test_url = "https://webwork.bridgew.edu/oer/Business_Calculus/ch-functions.html"
# test_url = "https://en.wikipedia.org/wiki/Mercedes_Sosa"
test_url = "https://webwork.bridgew.edu/oer/Business_Calculus/ch-combiningfns.html"
print("URL CONTENTS\n", robust_get_page_text(test_url))
# robust_get_page_text(test_url)

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

