import pandas as pd

data = {
  "age": [50, 40, 30, 40, 20, 10, 30],
  "qualified": [True, False, False, False, False, True, True]
}
df = pd.DataFrame(data)

newdf = df.where(df["age"] > 30).dropna()["age"]


print(newdf)
