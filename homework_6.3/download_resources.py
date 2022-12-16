import requests


URL = "http://share.yellowrobot.xyz/quick/2022-11-7-15633F14-6EA9-47CB-A75E-6C508C6AA470.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
