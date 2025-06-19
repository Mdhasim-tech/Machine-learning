import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

digits = load_digits()
X = digits.data       # shape: (1797, 64) — each image is flattened to 64 pixels
y = digits.target     # labels: digits 0 through 9
print(digits.images)
# plt.figure(figsize=(8, 4))
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(digits.images[i], cmap='gray')
#     plt.title(f"Label: {digits.target[i]}")
#     plt.axis('off')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


index = 10  # any sample index from test set
plt.imshow(digits.images[index], cmap='gray')
plt.title(f"Actual: {y_test[index]} | Predicted: {knn.predict([X_test[index]])[0]}")
plt.axis('off')
plt.show()