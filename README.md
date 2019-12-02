## Left to do 
 - Tree depth : 2 + 5 , add these to report 
 - Cross check tree and PRISM result for important attributes (AIT201 etc)
 - Rerun Weka stuff and screen shot settings + include interesting results
 - Gather accuracy of each weka test to pick for ensemble
 - Ensemble in r and python just to compare
  - Use the parsed association rules to figure out which conditions should be dropped (clopper pearson or y1 y2 e1 e2) (need to do decision tree pruning)
  
### Reorganization of models
- Clusterings - sujit
  - kmeans 
  - nearest neighbors
  - dbscan
- regression
- SVMs
- PRISM 
- Baysian Networks
- Ensemble


## NOTES FROM CLASS:

We need to use ensemble methods. Leopold also mentioned SVMs and Decision Trees.

Primarily interested in being able to detect attacks ("they didn't just attack one thing at a time; sometimes 2 or 3 things."

MV#0# values:
0 => in transition
1 => closed
2 => open

We may want to use unsupervised methods. Try clustering and association rules. Find patters in the rows where is_attack = 0 and see if they hold in the rows where is_attack = 1
In order to do association rule mining with continuous attributes, we have to first convert them to binary nomincal attributes. r can compute entropy and information gain.

The dataset can be cleaned more, but Leopold doesn't really think that we need to clean it more.
She warned that we should not get rid of columns that don't vary much in value.

Drop unnecessary conditions in your rules! "Get them as tight as possible." => clopper pearson for trees. Also maybe Y1 Y2 E1 E2

Wants a report in the end "These are the most important findings".
"Understanding what all of the idfferent pieces do helps with that."
"Does it make sense that it's this valve and this tank?"
"If water's flowing backwards, we have a problem."

Include Pie charts, clusters, and tables in the final report.
Also mention how many rows an association rule holds true in.

We can go over 2 pages if we included images (the 2 page limit is for text only).
"The 100 page pdf is your friend."
Include all source code.
