import numpy as np
import pandas as pd

data = pd.read_csv("marks.csv")
X = data[["Vize", "Final"]]
Y = data["Durum"]
Y = Y.replace("Failed", 0)
Y = Y.replace("Pass", 1)

x = np.array(X)
y = np.array(Y)