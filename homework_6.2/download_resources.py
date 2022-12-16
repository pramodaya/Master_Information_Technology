import requests


URL = "http://share.yellowrobot.xyz/quick/2022-11-4-CA07B56A-1BD9-4254-87CE-D798403B7B7A.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
