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
from bs4 import BeautifulSoup as bs
import codecs
import re
from webdriver_manager.chrome import ChromeDriverManager

import requests
from bs4 import BeautifulSoup

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 10)

prefix = "https://help.webex.com/en-us/"
with open(cwd+"/initial_url_list", "r+") as f:
    urls = f.readlines()
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
           if (href not in visited) and href.startswith(prefix):
                urls.append(href)

while (len(urls) > 0) and (len(visited) < 5000):
    print(len(urls), len(visited))
    current_url = urls.pop(0)
    driver.get(current_url)
    get_url = driver.current_url

    filename = get_url.removeprefix(prefix).replace("/","_")

    print(get_url, current_url)
    if cwd+"/"+filename in saved_pages:
        if get_url not in visited: visited.append(get_url)
        if len(urls) < 200: filter_and_add_urls(next_urls)
        print("page already saved: "+filename)
        continue

    body = driver.find_element(By.TAG_NAME, "body")
    if filename.startswith("landing"):
        next_urls = body.find_elements(By.CLASS_NAME, "landing-revamp_articleLink__KGIzU")
        filter_and_add_urls(next_urls)
        continue

    elif filename.startswith("article"):
        main_frame = body.find_element(By.CLASS_NAME, "article_article-left-content__1AUsB")
        main_text = main_frame.text
        next_urls = main_frame.find_elements(By.TAG_NAME, 'a')
        print(filename)
        with open(cwd+"/pages/"+filename, "w+") as file:
            file.write(main_text)

        if get_url not in visited: visited.append(get_url)
        filter_and_add_urls(next_urls)
    time.sleep(0.5)
