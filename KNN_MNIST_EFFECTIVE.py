import numpy as np
from mnist import MNIST
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 
# Load MNIST data
mndata = MNIST('C:\\Users\\nguye\\Downloads\\MNIST')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce dimensionality
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Optimize hyperparameters
param_grid = {
    'n_neighbors': np.arange(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}
knn = neighbors.KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train_pca, y_train)

print("Best parameters:", knn_cv.best_params_)

# Evaluate model
y_pred = knn_cv.predict(X_test_pca)
print("Accuracy of optimized KNN for MNIST: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
