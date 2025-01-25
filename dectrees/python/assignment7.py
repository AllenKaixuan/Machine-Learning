import random
import monkdata as m
import dtree as d
import matplotlib.pyplot as plt
import numpy as np


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune_tree(tree, validation):
    pruned_tree = d.allPruned(tree)
    best_tree = max(pruned_tree, key=lambda x: d.check(x, validation))
    if d.check(best_tree, validation) >= d.check(tree, validation):
        return prune_tree(best_tree, validation)
    else:
        return tree

def find_fraction(data, fraction):
    train, val = partition(data, fraction)
    tree = d.buildTree(train, m.attributes)
    pruned_tree = prune_tree(tree, val)
    return pruned_tree

def compute_statistics(data, testdata, fractions, n_runs=10):
    means = []
    stds = []
    
    for f in fractions:
        accuracies = []
        for _ in range(n_runs):
            pruned_tree = find_fraction(data, f)
            accuracies.append(d.check(pruned_tree, testdata))
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
    
    return means, stds

n_runs = 10
fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
Accuracy_1 = []
Accuracy_3 = []

# for f in fraction:
#     pruned_tree = find_fraction(m.monk1, f)
#     # print(f"Fraction {f}:")
#     # print(f"Accuracy: {d.check(pruned_tree, m.monk1test)}")
#     Accuracy_1.append(d.check(pruned_tree, m.monk1test))

# for f in fraction:
#     pruned_tree = find_fraction(m.monk3, f)
#     # print(f"Fraction {f}:")
#     # print(f"Accuracy: {d.check(pruned_tree, m.monk3test)}")
#     Accuracy_3.append(d.check(pruned_tree, m.monk3test))
means1, stds1 = compute_statistics(m.monk1, m.monk1test, fraction, n_runs)
means3, stds3 = compute_statistics(m.monk3, m.monk3test, fraction, n_runs)

# without pruning
tree_1 = d.buildTree(m.monk1, m.attributes)
accuracy_1 = d.check(tree_1, m.monk1test)  
print(accuracy_1)


tree_3 = d.buildTree(m.monk3, m.attributes)
accuracy_3 = d.check(tree_3, m.monk3test)
print(accuracy_3)

plt.figure(figsize=(12, 6))
plt.errorbar(fraction, means1, yerr=stds1, fmt='o-', label='Monk1')
plt.errorbar(fraction, means3, yerr=stds3, fmt='o-', label='Monk3')

plt.xlabel('Training Set Fraction')
plt.ylabel('Accuracy')
plt.title('Mean Accuracy vs Training Fraction (with standard deviation)')
plt.legend()
plt.grid(True)
plt.show()

# assignment 6
# pruning will reduce the complexity of the tree, which will reduce the overfitting, and increase the generalization ability of the tree
# the pruning is to reduce the variance and increase the bias


