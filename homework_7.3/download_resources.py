import requests


URL = "http://share.yellowrobot.xyz/quick/2022-11-25-2D6102EF-0B92-40A2-BE23-B1EEA2D3B120.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
