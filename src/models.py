from . import config
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

class ModelTrainer:
    """Train and evaluate ML models"""

    def __init__(self, id_to_label):
        self.id_to_label = id_to_label
        self.sorted_labels = [id_to_label[i] for i in sorted(id_to_label.keys())]

    def train_kmeans(self, X_train, y_train, X_test, y_test, n_clusters=None):
        n_clusters = n_clusters or config.KMEANS_CLUSTERS
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE)
        cluster_ids = kmeans.fit_predict(X_train)

        # Assign label to clusters
        cluster_to_label = {}
        for cluster_id in set(cluster_ids):
            labels_in_cluster = [y_train[i] for i in range(len(y_train)) if cluster_ids[i] == cluster_id]
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            cluster_to_label[cluster_id] = most_common_label

        # Predict
        test_cluster_ids = kmeans.predict(X_test)
        y_pred = [cluster_to_label[cluster_id] for cluster_id in test_cluster_ids]

        return self._evaluate(y_test, y_pred)
    
    def train_knn(self, X_train, y_train, X_test, y_test, n_neighbors=None):
        """Train and evaluate KNN classifier"""
        n_neighbors = n_neighbors or config.KNN_NEIGHBORS
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        return self._evaluate(y_test, y_pred)
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Decision Tree classifier"""
        dt = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        return self._evaluate(y_test, y_pred)
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Naive Bayes classifier"""
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)

        return self._evaluate(y_test, y_pred)
    
    def _evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.sorted_labels, output_dict=True)
        return y_pred, accuracy, report

def train_all_models(vectorized_data, y_train, y_test, id_to_label):
    """Train all models on all vectorization methods"""
    print("Training Models")

    trainer = ModelTrainer(id_to_label)
    results = {}

    model_funcs = {
        'kmeans': trainer.train_kmeans,
        'knn': trainer.train_knn,
        'decision_tree': trainer.train_decision_tree,
        'naive_bayes': trainer.train_naive_bayes
    }

    for model_name, model_func in model_funcs.items():
        print(f"\n{model_name.upper()}:")
        results[model_name] = {}
        
        for vec_name, (X_train, X_test) in vectorized_data.items():
            y_pred, accuracy, report = model_func(X_train, y_train, X_test, y_test)
            results[model_name][vec_name] = {
                'predictions': y_pred,
                'accuracy': accuracy,
                'report': report
            }
            print(f"  {vec_name}: {accuracy:.4f}")
    
    return results










