import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support, classification_report
import time
import cv2
from sklearn.svm import SVC  # SVM için eklenen kütüphane
import os
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score, classification_report
import time

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = img / 255.0
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return images
# load_images_from_folder ve diğer gerekli fonksiyonlarınız

classes = ['biotite', 'bornite', 'chrysocolla', 'malachite', 'muscovite', 'pyrite', 'quartz']
base_path = 'foldlar/'
results_path = 'SoftmaxResults/'

selected_models = ["alex", "VGG16", "Xception"]
classifiers = {
    'SVM Linear': SVC(kernel='linear', probability=True),
    'SVM Poly': SVC(kernel='poly', probability=True),
    'SVM RBF': SVC(kernel='rbf', probability=True),
    'SVM Sigmoid': SVC(kernel='sigmoid', probability=True),

}
for fold in range(1, 6):
    print(f"Processing Fold {fold}")
    X_train, y_train, X_test, y_test = [], [], [], []

    for class_label, class_name in enumerate(classes):
        train_folder = os.path.join(base_path, f'fold_{fold}/train/{class_name}')
        test_folder = os.path.join(base_path, f'fold_{fold}/test/{class_name}')
        train_images = load_images_from_folder(train_folder)
        test_images = load_images_from_folder(test_folder)
        X_train.extend(train_images)
        y_train.extend([class_label] * len(train_images))
        X_test.extend(test_images)
        y_test.extend([class_label] * len(test_images))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train_features_combined = None
    X_test_features_combined = None

    for md in selected_models:
        model_path = f'SoftmaxResults/{md}/fold_{fold}/{md}_fold_{fold}.hdf5'
        model = tf.keras.models.load_model(model_path)
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('gap_layer').output)

        X_train_features = intermediate_layer_model.predict(X_train)
        X_test_features = intermediate_layer_model.predict(X_test)

        if X_train_features_combined is None:
            X_train_features_combined = X_train_features
            X_test_features_combined = X_test_features
        else:
            X_train_features_combined = np.concatenate((X_train_features_combined, X_train_features), axis=1)
            X_test_features_combined = np.concatenate((X_test_features_combined, X_test_features), axis=1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features_combined)
    X_test_scaled = scaler.transform(X_test_features_combined)

    for classifier_name, classifier in classifiers.items():
        print(f"\n----- {classifier_name} Classifier, Fold {fold} -----")
        
        start_time_train = time.time()
        
        classifier.fit(X_train_scaled, y_train)
        
        end_time_train = time.time()
        training_time = end_time_train - start_time_train
        
        start_time_test = time.time()
        
        y_pred = classifier.predict(X_test_scaled)
        
        end_time_test = time.time()
        testing_time = end_time_test - start_time_test
        
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        auc = roc_auc_score(y_test, classifier.predict_proba(X_test_scaled), multi_class='ovr')
        cm = confusion_matrix(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        class_report = classification_report(y_test, y_pred, target_names=classes, digits=5)
        results_path="FusionRes"
        result_file_path = os.path.join(results_path, f"fold_{fold}", f"{classifier_name}.txt")
        with open(result_file_path, 'w') as file:
            file.write(f"Cohen's Kappa: {kappa}\n")
            file.write(f"AUC: {auc}\n")
            file.write(f"Training Time: {training_time:.10f} seconds\n")
            file.write(f"Testing Time: {testing_time:.10f} seconds\n")
            file.write("\nConfusion Matrix:\n")
            file.write(np.array2string(cm, separator=', '))
            file.write(f"\n\nPrecision (Weighted Avg): {precision}\n")
            file.write(f"Recall (Weighted Avg): {recall}\n")
            file.write(f"F1-Score (Weighted Avg): {f1_score}\n")
            file.write(f"Accuracy: {accuracy}\n")
            file.write("\n\nFull Classification Report:\n")
            file.write(classification_report(y_test, y_pred, target_names=classes, digits=7))
        
    print(f"Results saved to ")
            
