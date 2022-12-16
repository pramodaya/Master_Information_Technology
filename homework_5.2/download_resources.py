import requests


URL = "http://share.yellowrobot.xyz/quick/2022-10-28-54162E4D-E187-46D3-8C15-79373A7BD749.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
