

from collections import defaultdict
from math import floor

data = ''
with open("dataset.arff", "r") as h:
    data = h.read()

flag = False
aggregate = []
freq = defaultdict(int)
#minn = 99999999.0
#maxx = -999999999.0
ctr = 0
for i in data.split("\n"):
    if not i.strip():
        continue
    if ctr % 100000 == 0:
        print(ctr)
    ctr += 1
    if "@data" in i:
        flag = True
        continue
    if flag:
        spl = i.split(",")
        val = float(spl[20])
        freq[round(val,2)] += 1
        aggregate.append(val)

print("min:", min(aggregate))
print("max:", max(aggregate))
print("frequency analysis:", '\n'.join(f'{i}: {j}' for i,j in sorted(freq.items()))) 


