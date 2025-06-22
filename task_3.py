
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt


try:
   df = pd.read_csv(r'D:\Internship projects\Internship 1\task_3\bank+marketing\bank\bank-full.csv', delimiter=';')
   print(f"Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("File not found. Check the file path and make sure the file exists.")
    print("Current path being used: D:\\Internship projects\\Internship 1\\task_3\\bank+marketing\\bank\\bank-full.csv")
    exit()


X = df.drop('y', axis=1)  
y = df['y']  

print("Target distribution:")
print(y.value_counts())


label_encoders = {}
X_encoded = X.copy()


categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"Encoded {len(categorical_cols)} categorical columns")


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=3,  
    min_samples_split=100, 
    min_samples_leaf=50     
)

print("Training model...")
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\nGenerating decision tree visualization...")
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_classifier, 
               feature_names=X_encoded.columns,
               class_names=['no', 'yes'],
               filled=True,
               fontsize=12,
               rounded=True)
plt.title("Decision Tree Classifier - Bank Marketing (Depth=3)", fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDecision Tree Rules:")
for i, (feature, threshold, left, right) in enumerate(zip(
    dt_classifier.tree_.feature,
    dt_classifier.tree_.threshold,
    dt_classifier.tree_.children_left,
    dt_classifier.tree_.children_right
)):
    if left == right: 
        continue
    feature_name = X_encoded.columns[feature]
    print(f"Node {i}: If {feature_name} <= {threshold:.2f}")

print(f"\nTree has {dt_classifier.tree_.n_leaves} leaf nodes and depth of {dt_classifier.tree_.max_depth}")

print("\nDecision Tree Rules (first few levels):")
tree_rules = tree.export_text(dt_classifier,feature_names=list(X_encoded.columns),max_depth=4)
print(tree_rules[:2000]) 

print("\nModel training complete!")
print("Tree visualization saved as 'decision_tree_visualization.png'")