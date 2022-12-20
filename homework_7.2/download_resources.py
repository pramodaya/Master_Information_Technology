import requests


URL = "http://share.yellowrobot.xyz/quick/2022-11-25-F7AD4B22-60AE-4AF5-85DE-0FA4886D0001.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
