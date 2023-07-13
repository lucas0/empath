import glob
import time
import os,sys
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException, StaleElementReferenceException
from bs4 import BeautifulSoup
import codecs
import re
from webdriver_manager.chrome import ChromeDriverManager

import requests
from bs4 import BeautifulSoup

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

chrome_options = Options()
chrome_options.add_argument("--headless")

#service = Service(executable_path=cwd+"/chromedriver")

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 10)

start_url = 'https://www.cisco.com/c/en/us/products/contact-center/webex-contact-center-enterprise/index.html'
urls = [start_url]
visited = []
saved_pages = glob.glob(cwd+"/pages/*.html")

def filter_and_add_urls(next_urls):
    for e in next_urls:
        try:
            href = e.get_attribute('href')
        except (StaleElementReferenceException, WebDriverException) as error:
            print(error)
            continue
        if (href is not None):
           href = href.split(".html")[0]+".html"
           if (href not in visited) and \
              (href.startswith("https://www.cisco.com/c/en/us")) and \
              (href != "https://www.cisco.com/site/us/en/index.html"):
                urls.append(href)

while (len(urls) > 0) and (len(visited) < 5000):
    print(len(urls), len(visited))
    current_url = urls.pop(0)
    driver.get(current_url)
    get_url = driver.current_url
    next_urls = driver.find_elements(By.TAG_NAME, 'a')

    print(get_url, current_url)
    filename = get_url.removeprefix("https://www.cisco.com/c/en/us/").replace("/","_")
    if "/Users/lucazeve/Coding/scrapper/pages/"+filename in saved_pages:
        if get_url not in visited: visited.append(get_url)
        if len(urls) < 200: filter_and_add_urls(next_urls)
        print("page already saved: "+filename)
        continue

    try:
        wait.until(EC.url_to_be(current_url))
    except TimeoutException as ex:
        if current_url not in visited: visited.append(current_url)
        print(str(ex)+get_url)
        continue


    body = driver.find_element(By.TAG_NAME, "body")
    main_text = body.find_element(By.ID, "fw-content").text
    print(filename)
    with open(cwd+"/pages/"+filename, "w+") as file:
        file.write(main_text)

    if get_url not in visited: visited.append(get_url)
    if len(urls) < 500: filter_and_add_urls(next_urls)
    random.shuffle(urls)
    time.sleep(0.5)
