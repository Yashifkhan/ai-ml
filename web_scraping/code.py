import requests

url="https://www.scrapethissite.com/pages/simple/"
resp=requests.get(url)
# print(resp.status_code)

if( resp.status_code == 200):
    with open("web_sriping_data.html","w",encoding="utf-8") as f :
        f.write(resp.text)   
