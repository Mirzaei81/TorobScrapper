from random import randint
import numpy as np 
import pandas as pd
df = pd.DataFrame({"val":np.random.rand(10)*5,"locs":["T" if randint(0,1)==1 else "nT" for _ in range(10)]})
df["target"] = np.where(df["locs"]=="T",1.4,1.2)

print(df)