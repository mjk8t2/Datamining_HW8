import pandas as pd
import itertools

df = pd.read_csv("../binNomified.csv")

def get_assoc_rule(attrs):
  curdf = df
  for attr, val in attrs:
    curdf = curdf.loc[(df[attr] == val)]
      # curdf = curdf.loc
  support_yes = len(curdf.loc[(df["is_attack"] == "one")].index)
  conf_yes = support_yes/len(curdf.index) if len(curdf.index) != 0 else 0
  return (conf_yes, support_yes)

values = [(column, val) for column in df for val in df[column].unique() if column != "is_attack"]
maxdepth = 4
print("building conditions...")

combs = []
for ii in range(maxdepth):
  combs1 = [comb for comb in itertools.combinations(values, ii+1)]
  combs.extend(combs1)

print("looking for association rules...")

# this takes about 6 hours to run
best_confidence = 0
data = {"Rule":[], "Conf_yes":[], "Numrowstrue":[], "Outof":[]}
for ii, val in enumerate(combs):
  conf_yes, support_yes = get_assoc_rule(val)
  infostr = " && ".join(["{}={}".format(col, val) for col, val in val])
  data["Rule"].append(infostr)
  data["Conf_yes"].append(conf_yes)
  data["Numrowstrue"].append(support_yes)
  if conf_yes != 0:
    data["Outof"].append(int(support_yes/conf_yes))
  else:
    data["Outof"].append(0)

  if conf_yes > 0.9 and support_yes > 700:
    infostr = " && ".join(["{}={}".format(col, val) for col, val in val]) + " => conf_yes={} in {}/{} rows".format(conf_yes, support_yes, int(support_yes/conf_yes))
    print(infostr)
  
  if (ii + 1) % 200 == 0:
    posi = "{}/{}".format(ii + 1, len(combs))
    print(posi, end='')
    print('\b' * len(posi), end='', flush=True)

df2 = pd.DataFrame.from_dict(data)
df2.to_csv("allrules.csv")



# import random

# datafile = open("goodrules.txt", 'w')

# seed = random.randrange(2**32 - 1) # https://stackoverflow.com/a/5012617
# # seed = 3092108368
# print("Seed was:", seed)
# random.seed(seed)

# datafile.write("Seed: {}\n".format(seed))




# values = [(column, val) for column in df for val in df[column].unique() if column != "is_attack"]

# # numconditions = 5000
# maxdepth = 4
# print("building conditions...")

# combs = []
# for ii in range(maxdepth):
#   combs1 = [comb for comb in itertools.combinations(values, ii+1)]
#   combs.extend(combs1)

# print("looking for association rules...")

# # this takes about 6 hours to run
# best_confidence = 0
# for ii, val in enumerate(combs):
#   conf_yes, support_yes = get_assoc_rule(val)
#   if conf_yes > 0.9 and support_yes > 700 and conf_yes > best_confidence:
#     infostr = " && ".join(["{}={}".format(col, val) for col, val in val]) + " => conf_yes={} in {}/{} rows".format(conf_yes, support_yes, int(support_yes/conf_yes))
#     datafile.write(infostr + "\n")
#     print(infostr)
#     best_confidence = conf_yes
  
#   if (ii + 1) % 200 == 0:
#     posi = "{}/{}".format(ii + 1, len(combs))
#     print(posi, end='')
#     print('\b' * len(posi), end='', flush=True)


# datafile.close()