import pandas

df = pandas.read_csv("allrules.csv")


print("df loccing")
df = df.loc[ (df["Conf_yes"] > 0.9) & (df["Numrowstrue"] > 400) ]

print("genning list")
df['numconditions'] = [4 for ii in range(len(df.index))]
for index, row in df.iterrows():
  df.loc[index,"numconditions"] = len(row["Rule"].split(" && "))

print("sorting df")
df = df.sort_values(by=['numconditions', 'Conf_yes', 'Numrowstrue'], ascending=[True, False, False])

print(df)
df.to_csv("interestingrules.csv")