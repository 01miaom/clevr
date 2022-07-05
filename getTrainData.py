import requests
import bs4

url=""
response = requests.get(url)
data=bs4.BeautifulSoup(response.content, "html.parser")
element = data.find_all("img", class_="mimg")
for i in range(1,100):
    response = requests.get(element[i].get('src'))
    fileName = str(i)+'.jpeg'
    with open(fileName.format(0),'wb') as f:
        f.write(response.content)
