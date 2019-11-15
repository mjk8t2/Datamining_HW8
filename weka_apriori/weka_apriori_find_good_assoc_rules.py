import re

"""
This script finds all association rules with one antecedent and one consequent
"""

with open("apriori_output.txt") as f:
  data = f.read()
    
    
outstr = ""
expression = r'([0-9]+\. .*?is_attack=one.*? ==> .*)'
expression = r'([0-9]+\. .*? ==> .*?is_attack=one.*)'
for match in re.finditer(expression,data):
  # print(match.group(1))
  outstr += match.group(1) + "\n"
  
with open("good_rules.txt", 'w') as f:
  f.write(outstr)
