import requests


URL = "http://share.yellowrobot.xyz/quick/2022-12-2-F9A7F058-B405-41A4-81E1-26D1B115197F.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
