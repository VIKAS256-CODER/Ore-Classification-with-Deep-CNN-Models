import os
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from mealpy import BinaryVar, FPA, ABC, GA, PSO, MA

log_folder = "FusionFeatureSelection"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
iter = 0

def log_results(fold, vector, accuracy, precision, recall, f1_score, auc_score, kappa_score, conf_matrix, is_average=False):
    global iter
    iter += 1
    prefix = 'mean_' if is_average else ''
    log_filename = os.path.join(log_folder, f'{prefix}iter{iter}_fold_{fold}.txt')
    with open(log_filename, 'w') as file:
        file.write(f"iter: {iter}, Fold: {fold}\n")
        file.write("Vector: [" + ", ".join(map(str, vector)) + "]\n")
        file.write(f"{prefix.capitalize()}Accuracy: {accuracy:.5f}\n")
        file.write(f"{prefix.capitalize()}Precision: {precision:.5f}\n")
        file.write(f"{prefix.capitalize()}Recall: {recall:.5f}\n")
        file.write(f"{prefix.capitalize()}F1 Score: {f1_score:.5f}\n")
        file.write(f"{prefix.capitalize()}AUC Score: {auc_score:.5f}\n")
        file.write(f"{prefix.capitalize()}Cohen's Kappa: {kappa_score:.5f}\n")
        file.write(f"{prefix.capitalize()}Confusion Matrix:\n{conf_matrix}\n")

def load_and_preprocess_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = img / 255.0
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return images

def extract_features(model, images):
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('gap_layer').output)
    return intermediate_layer_model.predict(images)

def calc(feature_mask):
    total_accuracy, total_precision, total_recall, total_f1_score, total_auc, total_kappa = 0, 0, 0, 0, 0, 0
    sum_conf_matrix = None
    models = ["alex", "VGG16", "Xception"]
    for fold in range(1, 6):
        X_train, y_train = [], []
        X_test, y_test = [], []

        for class_label, class_name in enumerate(classes):
            train_folder = os.path.join(base_path, f'fold_{fold}/train/{class_name}')
            test_folder = os.path.join(base_path, f'fold_{fold}/test/{class_name}')

            train_images = load_and_preprocess_images(train_folder)
            test_images = load_and_preprocess_images(test_folder)

            X_train.extend(train_images)
            y_train.extend([class_label] * len(train_images))
            X_test.extend(test_images)
            y_test.extend([class_label] * len(test_images))

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        X_train_features = np.hstack([extract_features(tf.keras.models.load_model(f'SoftmaxResults/{md}/fold_{fold}/{md}_fold_{fold}.hdf5'), X_train) for md in models])
        X_test_features = np.hstack([extract_features(tf.keras.models.load_model(f'SoftmaxResults/{md}/fold_{fold}/{md}_fold_{fold}.hdf5'), X_test)])
        X_train_masked = X_train_features * feature_mask
        X_test_masked = X_test_features * feature_mask

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_masked)
        X_test_scaled = scaler.transform(X_test_masked)

        classifier = SVC(kernel='linear', probability=True)
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        y_pred_proba = classifier.predict_proba(X_test_scaled)

        # Metriklerin hesaplanması
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        kappa = cohen_kappa_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        total_accuracy += acc
        total_precision += prec
        total_recall += rec
        total_f1_score += f1
        total_auc += auc
        total_kappa += kappa

        if sum_conf_matrix is None:
            sum_conf_matrix = conf_matrix
        else:
            sum_conf_matrix += conf_matrix

        log_results(fold, feature_mask, acc, prec, rec, f1, auc, kappa, conf_matrix)

    mean_accuracy = total_accuracy / 5
    mean_precision = total_precision / 5
    mean_recall = total_recall / 5
    mean_f1_score = total_f1_score / 5
    mean_auc = total_auc / 5
    mean_kappa = total_kappa / 5
    mean_conf_matrix = sum_conf_matrix / 5

    log_results(0, feature_mask, mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_auc, mean_kappa, mean_conf_matrix, is_average=True)

    return mean_f1_score

problem_dict = {
    "bounds": BinaryVar(n_vars=2816, name="delta"),
    "minmax": "max",
    "obj_func": calc
}

# List of optimization algorithms to apply
optimizers = {
    "FPA": FPA.OriginalFPA(epoch=100, pop_size=50),
    "ABC": ABC.OriginalABC(epoch=100, pop_size=50),
    "GA": GA.OriginalGA(epoch=100, pop_size=50),
    "PSO": PSO.OriginalPSO(epoch=100, pop_size=50),
    "Memetic": MA.BaseMA(epoch=100, pop_size=50)
}

# Running each optimizer and logging results
for name, optimizer in optimizers.items():
    print(f"Running {name} algorithm...")
    model = optimizer.solve(problem_dict)
    print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
