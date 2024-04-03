# This file is used to prune the dataset to only 1000 lines for simplicity sake


content = b""
with open("../dataset/dataset.arff", "rb") as h:
    buff = ""
    while b"@data" not in (buff := h.readline()):
        content += buff
    content += buff
    for i in range(1000): # 1000 lines
        for _ in range(8000):
            h.readline()
        buff = h.readline()
        content += buff

with open("../dataset/pruned.arff", "wb") as h:
    h.write(content)

print("should be fine")

