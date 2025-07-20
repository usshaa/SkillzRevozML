import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Sample data (age and income)
X = np.array([[22, 25000], [28, 45000], [45, 70000], [52, 90000], [65, 120000]])
y = np.array(['Yes', 'Yes', 'Yes', 'No', 'No'])

# Train a Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=5, random_state=42)
rf_clf.fit(X, y)

# Save each individual decision tree from the Random Forest
for i, estimator in enumerate(rf_clf.estimators_):
    # Create a plot for each tree
    plt.figure(figsize=(10,8))
    tree.plot_tree(estimator, filled=True, feature_names=['Age', 'Income'], class_names=['No', 'Yes'])
    
    # Save each tree plot as a PNG image
    plt.savefig(f"tree{i+1}_plot.png")
    plt.close()

print("All decision tree images are saved successfully.")