import requests
import bs4

count = 20
searchName = 'cylinder'
url="https://cn.bing.com/images/async?q="+searchName+"&first=12&count="+str(count)+"&cw=1177&ch=729&relp=35&tsc=ImageHoverTitle&datsrc=I&layout=RowBased&mmasync=1"
headers = {
    'cookie': '', #fill with your cookie
    'user-agent': '' #fill with your user-agent
}

response = requests.get(url, headers=headers)
data=bs4.BeautifulSoup(response.content, "html.parser")
element = data.find_all("img", class_="mimg")
for i in range(1,count):
    response = requests.get(element[i].get('src'))
    fileName = str(i)+'.jpeg'
    with open(fileName.format(0),'wb') as f:
        f.write(response.content)
