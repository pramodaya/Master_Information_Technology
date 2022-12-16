import numpy as np
import requests

# a = np.arange(12).reshape(4,3)
# b = np.arange(12).reshape(3,4)

a = [
    [1, 2, 3],
    [4,5,6]
]

b = [
    [1,2],
    [3,4],
    [5,6]
]

print(a)
print("-------------")
print(b)
print("------DOT-------")
c = np.dot(a,b)
print(c)


URL = "http://share.yellowrobot.xyz/quick/2022-11-4-CA07B56A-1BD9-4254-87CE-D798403B7B7A.zip"
response = requests.get(URL)
open("data.zip", "wb").write(response.content)
