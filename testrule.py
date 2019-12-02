import numpy as np
import pandas

df = pandas.read_csv("hw8_data.csv")

# print(df)


litait = df.loc[(df["LIT301"] > 923.012) & (df["AIT203"] < 235.863)]
litait2 = df.loc[(df["LIT301"] > 923.012) & (df["AIT203"] < 235.863) & df["is_attack"] == 1]
aitait = df.loc[(df["AIT201"] < 128.509) & (df["AIT202"] < 8.934)]
# litaitbig = df.loc[(df["LIT301"] < 829) & (df["AIT203"] > 242.2)]
# litaitbigyes = df.loc[(df["LIT301"] < 829) & (df["AIT203"] > 242.2) & (df["is_attack"] == 0)]

both = pandas.merge(litait, aitait, how='inner', left_index=True, right_index=True)

print(len(litait))
print(len(litait2))
print(len(aitait))
# print(len(litaitbig), 'with yes:', len(litaitbigyes))
print(len(both))

# litaitbig.to_csv("confused.csv")

# litait.to_csv('litait.csv')
# aitait.to_csv('aitait.csv')
# both.to_csv("okkkkk.csv")


# print(len(aitait))
# kel = df.loc[(df["LIT301"] > 923) & (df["AIT203"] < 235.863) & (df["is_attack"] == 0)]

# print(len(bob))
# print(len(kel))

# for index, row in df.iterrows():
  # if row["LIT301"] > 925 and row["AIT203"] < 8.934:
    # if row["is_attack"] == 