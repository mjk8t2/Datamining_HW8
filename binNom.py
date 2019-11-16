import pandas as pd
import math

# C4.5 (part 1)_redVersion

df = pd.read_csv("hw8_data.csv")

df_discrete_colnames = [column for column in df if df.nunique()[column] < 5]
df_continuous_colnames = [column for column in df if df.nunique()[column] >= 5]
# df_discrete = df[df_discrete_colnames] # these are currently numeric
# df_continuous = df[df_continuous_colnames].astype(float)

def compute_entropy(partition, dff, nam):
  tot = len(dff.index)
  num_lt = len(dff.loc[df[nam] < partition].index)
  num_gt = len(dff.loc[df[nam] > partition].index)
  num_lt_natk = len(dff.loc[(df["is_attack"] == 0) & (df[nam] < partition)].index)
  num_lt_yatk = len(dff.loc[(df["is_attack"] == 1) & (df[nam] < partition)].index)
  num_gt_natk = len(dff.loc[(df["is_attack"] == 0) & (df[nam] > partition)].index)
  num_gt_yatk = len(dff.loc[(df["is_attack"] == 1) & (df[nam] > partition)].index)

  # print(num_lt, num_gt, num_lt_natk, num_lt_yatk, num_gt_natk, num_gt_yatk)

  entropy_lt_n = (num_lt_natk/num_lt)*math.log(num_lt_natk/num_lt, 2) if num_lt_natk != 0 else 0
  entropy_lt_y = (num_lt_yatk/num_lt)*math.log(num_lt_yatk/num_lt, 2) if num_lt_yatk != 0 else 0
  entropy_lt = -entropy_lt_n - entropy_lt_y

  entropy_gt_n = (num_gt_natk/num_gt)*math.log(num_gt_natk/num_gt, 2) if num_gt_natk != 0 else 0
  entropy_gt_y = (num_gt_yatk/num_gt)*math.log(num_gt_yatk/num_gt, 2) if num_gt_yatk != 0 else 0
  entropy_gt = -entropy_gt_n - entropy_gt_y

  entropyAfterSplit = (num_lt/tot)*entropy_lt + (num_gt/tot)*entropy_gt

  return entropyAfterSplit



for column in df_continuous_colnames:
  entropyAfterSplits = {}
  df_sorted = df.sort_values(by=[column])

  proc_freq = df_sorted[column].value_counts().sort_index()
  ndistinct = 0
  prev_val = -56435.4897651
  # compute entropy for all possible partitions
  for index, row in df_sorted.iterrows():
    xval = row[column]
    if prev_val != xval:
      ndistinct += 1

    # for each partition value that isn't on the edges...
    if ndistinct > 2 and ndistinct <= len(proc_freq.index) - 1 and prev_val != xval:
      partition = (prev_val + xval)/2
      entropyAfterSplits[partition] = compute_entropy(partition, df_sorted, column)
    elif prev_val != xval:
      pass
    prev_val = xval

  best_partition = sorted(entropyAfterSplits.items(), key=lambda x: x[1], reverse=False)[0][0] # lowest entropyAfterSplit => highest infogain

  print(column, best_partition)
  keynum = -10056405641324.65465 # no recorded value is smaller than this number
  df.loc[df[column] < best_partition,column] = keynum
  df.loc[df[column] > best_partition,column] = "GT_{}".format(best_partition)
  df = df.replace(keynum, "LT_{}".format(best_partition))

df = df.replace(0, "zero").replace(1, "one").replace(2, "two") # make them nominal for weka
df.to_csv("binNomified.csv",index=False)