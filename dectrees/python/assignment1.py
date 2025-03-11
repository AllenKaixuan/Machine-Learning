import monkdata as m
import dtree as d


print("Entropy of monk1: ", d.entropy(m.monk1test)) # 1.0
print("Entropy of monk2: ", d.entropy(m.monk2test)) # 0.957117428264771
print("Entropy of monk3: ", d.entropy(m.monk3test)) # 0.9998061328047111



# assignment 2
# like coins and die, they are uniform distribution, the entrophy is log2(n), n is the size of sample space

# non-uniform distribution will have smaller entropy, because it is more  predictable