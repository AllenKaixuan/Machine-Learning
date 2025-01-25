import dtree as d
import monkdata as m
import drawtree_qt5 as draw


gains = [d.averageGain(m.monk1, attr) for attr in m.attributes]

# returns the maximum value in the list of gains
best_attr_index = gains.index(max(gains))
print(f"Best attribute for splitting: Attribute {best_attr_index + 1}")

# choose the best attribute for splitting
best_attr = m.attributes[best_attr_index]
subsets = [d.select(m.monk1, best_attr, value) for value in best_attr.values]

# calculate the information gain for each subset except the one used for splitting before
for i, subset in enumerate(subsets):
    print(f"Subset {i + 1}:")
    for j, attr in enumerate(m.attributes):
        if j == best_attr_index:
            continue
        gain = d.averageGain(subset, attr)
        print(f"  Information Gain for Attribute {j + 1}: {gain}")

# calculate the majority class for each subset
for i, subset in enumerate(subsets):
    majority_class = d.mostCommon(subset)
    print(f"Majority class for Subset {i + 1}: {majority_class}")



# depth 2
tree_depth_2 = d.buildTree(m.monk1, m.attributes, 2)
print(tree_depth_2)


tree_1 = d.buildTree(m.monk1, m.attributes)
tree_2 = d.buildTree(m.monk2, m.attributes)
tree_3 = d.buildTree(m.monk3, m.attributes)


train_accuracy_1 = d.check(tree_1, m.monk1)
test_accuracy_1 = d.check(tree_1, m.monk1test)

train_accuracy_2 = d.check(tree_2, m.monk2)
test_accuracy_2 = d.check(tree_2, m.monk2test)

train_accuracy_3 = d.check(tree_3, m.monk3)
test_accuracy_3 = d.check(tree_3, m.monk3test)


train_error_1 = 1 - train_accuracy_1
test_error_1 = 1 - test_accuracy_1

train_error_2 = 1 - train_accuracy_2
test_error_2 = 1 - test_accuracy_2

train_error_3 = 1 - train_accuracy_3
test_error_3 = 1 - test_accuracy_3


print(f"MONK-1 Train Error: {train_error_1 * 100:.2f}%")
print(f"MONK-1 Test Error: {test_error_1 * 100:.2f}%")

print(f"MONK-2 Train Error: {train_error_2 * 100:.2f}%")
print(f"MONK-2 Test Error: {test_error_2 * 100:.2f}%")

print(f"MONK-3 Train Error: {train_error_3 * 100:.2f}%")
print(f"MONK-3 Test Error: {test_error_3 * 100:.2f}%")

draw.drawTree(tree_1)

#MONK-1 Train Error: 0.00%
#MONK-1 Test Error: 17.13%
#MONK-2 Train Error: 0.00%
#MONK-2 Test Error: 30.79%
#MONK-3 Train Error: 0.00%
#MONK-3 Test Error: 5.56%

#MONK-3 has the lowest test error, which means it is the best tree，because for the first node it has the highest information gain
#MONK-1 has the lowest test error, which means it is the best tree, because for the first node it has the lowest information gain