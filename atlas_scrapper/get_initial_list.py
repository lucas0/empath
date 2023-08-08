from bs4 import BeautifulSoup as bs
import os,sys

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

root = "https://help.webex.com"

with open(cwd+'/raw_html.html', 'r') as f:
    contents = f.read()
    soup = bs(contents)

#items = soup.find_all('div', {'class' : 'result_result-item__GJLqg'})
items = soup.find_all('li', {'class' : 'result_category-title__2HeIZ'})
items += soup.find_all('div', {'class' : 'result_result-heading__1Zx_V'})

links = list(set([root+i.find("a", href=True)["href"].split("#")[0]  for i in items]))

print(len(links))

with open(cwd+"/initial_url_list", "w+") as f:
    for i in links:
        f.write(i+"\n")
