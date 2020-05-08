import numpy as np
import pandas as pd

names = np.load("submission/names.npy")
models = [
    "submission/exp4_fold0.npy",
]

output = 0
for m in models:
    output += np.load(m) / len(models)
# print(output) 

data = []
data_with_extension = []
# print(type(output))
for n,o in zip(names, output):
    # print(o)
    # o /= np.sum(o)
    # data.append([n.split(".")[0]] + list(o))
    data_with_extension.append([n]+[o])
# print(data_with_extension)
df = pd.DataFrame(data=data_with_extension, columns=["Id", "Label"])
# df_ext = pd.DataFrame(data=data_with_extension, columns=["ID", "leaf_rust", "stem_rust", "healthy_wheat"])

df.to_csv("submission/submission.csv", index=False)
# df_ext.to_csv("submission/submission_ext.csv", index=False)
