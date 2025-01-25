import dtree as d
import monkdata as m
import pandas as pd

a1_monk_1 = d.averageGain(m.monk1, m.attributes[0])
a2_monk_1 = d.averageGain(m.monk1, m.attributes[1])
a3_monk_1 = d.averageGain(m.monk1, m.attributes[2])
a4_monk_1 = d.averageGain(m.monk1, m.attributes[3])
a5_monk_1 = d.averageGain(m.monk1, m.attributes[4])
a6_monk_1 = d.averageGain(m.monk1, m.attributes[5])

a1_monk_2 = d.averageGain(m.monk2, m.attributes[0])
a2_monk_2 = d.averageGain(m.monk2, m.attributes[1])
a3_monk_2 = d.averageGain(m.monk2, m.attributes[2])
a4_monk_2 = d.averageGain(m.monk2, m.attributes[3])
a5_monk_2 = d.averageGain(m.monk2, m.attributes[4])
a6_monk_2 = d.averageGain(m.monk2, m.attributes[5])

a1_monk_3 = d.averageGain(m.monk3, m.attributes[0])
a2_monk_3 = d.averageGain(m.monk3, m.attributes[1])
a3_monk_3 = d.averageGain(m.monk3, m.attributes[2])
a4_monk_3 = d.averageGain(m.monk3, m.attributes[3])
a5_monk_3 = d.averageGain(m.monk3, m.attributes[4])
a6_monk_3 = d.averageGain(m.monk3, m.attributes[5])

data = {
    'Attribute 1': [a1_monk_1, a1_monk_2, a1_monk_3],
    'Attribute 2': [a2_monk_1, a2_monk_2, a2_monk_3],
    'Attribute 3': [a3_monk_1, a3_monk_2, a3_monk_3],
    'Attribute 4': [a4_monk_1, a4_monk_2, a4_monk_3],
    'Attribute 5': [a5_monk_1, a5_monk_2, a5_monk_3],
    'Attribute 6': [a6_monk_1, a6_monk_2, a6_monk_3]
}

df = pd.DataFrame(data, index=['MONK-1', 'MONK-2', 'MONK-3'])
print(df)

print(d.bestAttribute(m.monk1, m.attributes))   # a5
print(d.bestAttribute(m.monk2, m.attributes))   # a5
print(d.bestAttribute(m.monk3, m.attributes))   # a2

#             Attribute 1  Attribute 2  Attribute 3  Attribute 4  Attribute 5  Attribute 6
#MONK-1       0.075273     0.005838     0.004708     0.026312     0.287031     0.000758
#MONK-2       0.003756     0.002458     0.001056     0.015664     0.017277     0.006248
#MONK-3       0.007121     0.293736     0.000831     0.002892     0.255912     0.007077
# choose attribute 5 for monk1, attribute 5 for monk2, attribute 2 for monk3

# assignment 4
# When the gain information is maximized, the entropy is minimized, which means more predictable 