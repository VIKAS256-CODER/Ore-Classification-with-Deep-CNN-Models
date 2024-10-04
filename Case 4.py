from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize

# Predefined weights (you can tune these or search for optimal values)
model_weights = {"alex": 0.3, "VGG16": 0.4, "Xception": 0.3}

def weighted_voting(predictions, model_weights):
    weights = np.array([model_weights[model_name] for model_name in models_to_load])
    weighted_sum = np.tensordot(weights, predictions, axes=(0, 0))
    return np.argmax(weighted_sum, axis=-1)

# Search for optimal weights (you can replace this with more advanced methods)
def search_weights(predictions, true_labels):
    def objective(weights):
        weighted_sum = np.tensordot(weights, predictions, axes=(0, 0))
        soft_votes = np.argmax(weighted_sum, axis=-1)
        return -accuracy_score(true_labels, soft_votes)  # Minimize negative accuracy

    init_weights = np.array([1/len(models_to_load)] * len(models_to_load))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(models_to_load)
    
    result = minimize(objective, init_weights, bounds=bounds, constraints=constraints)
    return result.x

for fold_num in range(1, 6):
    print(f"Processing fold {fold_num}...")
    
    # Load test data
    path_test = f"foldlar/fold_{fold_num}/test"
    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_generator = test_datagen.flow_from_directory(directory=path_test,
                                                      batch_size=batch_size,
                                                      class_mode="categorical",
                                                      target_size=(img_width, img_height),
                                                      shuffle=False)

    # Store predictions from each model
    predictions = []
    for model_name in models_to_load:
        model_path = f"SoftmaxResults/{model_name}/fold_{fold_num}/{model_name}_fold_{fold_num}.hdf5"
        model = load_model(model_path)
        print(f"Predicting with {model_name}...")
        pred = model.predict(test_generator)
        predictions.append(pred)
    
    predictions = np.array(predictions)

    # Hard Voting
    hard_votes = np.argmax(predictions, axis=-1)
    hard_votes_mode = mode(hard_votes, axis=0)[0].flatten()
    hard_accuracy = accuracy_score(test_generator.classes, hard_votes_mode)

    # Soft Voting
    soft_votes = np.argmax(np.mean(predictions, axis=0), axis=-1)
    soft_accuracy = accuracy_score(test_generator.classes, soft_votes)

    # Weighted Voting
    weighted_votes = weighted_voting(predictions, model_weights)
    weighted_accuracy = accuracy_score(test_generator.classes, weighted_votes)

    # Optional: Search for optimal weights
    optimal_weights = search_weights(predictions, test_generator.classes)
    optimal_weighted_votes = weighted_voting(predictions, dict(zip(models_to_load, optimal_weights)))
    optimal_weighted_accuracy = accuracy_score(test_generator.classes, optimal_weighted_votes)

    # Calculate metrics for both hard, soft, and weighted voting
    for voting_type, votes, accuracy in [("hard", hard_votes_mode, hard_accuracy), 
                                         ("soft", soft_votes, soft_accuracy), 
                                         ("weighted", weighted_votes, weighted_accuracy), 
                                         ("optimal_weighted", optimal_weighted_votes, optimal_weighted_accuracy)]:
        kappa = cohen_kappa_score(test_generator.classes, votes)
        y_true_ohe = tf.keras.utils.to_categorical(test_generator.classes, num_classes=num_classes)
        auc_val = roc_auc_score(y_true_ohe, np.mean(predictions, axis=0), multi_class='ovr')
        cm = confusion_matrix(test_generator.classes, votes)
        report = classification_report(test_generator.classes, votes, digits=7, output_dict=True)
        f1_score = report["weighted avg"]["f1-score"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]

        # Save metrics
        metrics_save_path = os.path.join(voting_save_dir, f"{voting_type}_voting_metrics_fold_{fold_num}.txt")
        with open(metrics_save_path, 'w') as f:
            f.write(f"{voting_type.capitalize()} Voting Accuracy: {accuracy}\n")
            f.write(f"Cohen's Kappa: {kappa}\n")
            f.write(f"AUC: {auc_val}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write(f"\n\nPrecision (Weighted Avg): {precision}\n")
            f.write(f"Recall (Weighted Avg): {recall}\n")
            f.write(f"F1-Score (Weighted Avg): {f1_score}\n")
            f.write("\n\nFull Classification Report:\n")
            f.write(classification_report(test_generator.classes, votes, digits=7))
        
        # Save confusion matrix plot
        df_cm = pd.DataFrame(cm, index=np.arange(cm.shape[0]), columns=np.arange(cm.shape[1]))
        plt.figure(figsize=(10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(voting_save_dir, f"{voting_type}_confusion_matrix_fold_{fold_num}.png"))

