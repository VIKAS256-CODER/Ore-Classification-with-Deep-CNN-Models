import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import time

batch_size = 32
img_width = 224
img_height = 224
epochs = 50  # Number of epochs for training
learning_rate = 0.0001  # Learning rate for Adam optimizer
iterations = 100  # Number of iterations for the optimization process

models = {
         "VGG16": tf.keras.applications.VGG16,
"MobileNet":tf.keras.applicatons.MobileNet
#..
#..
#..
#..
#Deep Models
#..

}

base_save_dir = Results
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

for fold_num in range(1, 6):
    data_path = f'foldlar/fold_{fold_num}'
    path_train = f"{data_path}/train"
    path_test = f"{data_path}/test"
    path_val = f"{data_path}/test"

    # Augmenting the training data to synthetically increase dataset size
    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=10,  # Small rotations
        width_shift_range=0.1,  # Horizontal shifting
        height_shift_range=0.1,  # Vertical shifting
        shear_range=0.2,  # Shearing
        zoom_range=0.2,  # Zooming
        brightness_range=[0.8, 1.2],  # Brightness adjustment
        horizontal_flip=True,  # Horizontal flipping
        fill_mode='nearest'  # Filling missing pixels
    )
    
    train_generator = train_datagen.flow_from_directory(
        directory=path_train,
        batch_size=64,
        class_mode="categorical",
        target_size=(img_width, img_height)
    )

    # Validation and test sets will not be augmented
    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_generator = test_datagen.flow_from_directory(
        directory=path_test,
        batch_size=64,
        class_mode="categorical",
        target_size=(img_width, img_height),
        shuffle=False
    )

    valid_datagen = ImageDataGenerator(rescale=1./255.)
    valid_generator = valid_datagen.flow_from_directory(
        directory=path_val,
        batch_size=64,
        class_mode="categorical",
        target_size=(img_width, img_height)
    )

    for model_name, model in models.items():
        print(f"Training {model_name} on fold {fold_num}...")
        base_model = model(include_top=False, input_shape=(img_width, img_height, 3))
        
        x = GlobalAveragePooling2D(name="gap_layer")(base_model.output)
        outputs = Dense(3, activation='softmax')(x)
        final_model = keras.Model(inputs=base_model.input, outputs=outputs)

        # Using Adam optimizer with learning rate of 0.0001 as specified
        final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        # Start training
        start_train_time = time.time()
        history = final_model.fit(train_generator,
                                  epochs=epochs,
                                  validation_data=valid_generator,
                                  verbose=1)
        train_duration = time.time() - start_train_time  # End timing

        # Start testing
        start_test_time = time.time()
        y_pred_prob = final_model.predict(test_generator)
        test_duration = time.time() - start_test_time  # End timing

        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true_ohe = tf.keras.utils.to_categorical(test_generator.classes, num_classes=3)

        save_dir = os.path.join(base_save_dir, f"{model_name}/fold_{fold_num}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_save_path = os.path.join(save_dir, f"{model_name}_fold_{fold_num}.hdf5")
        final_model.save(model_save_path)

        metrics_save_path = os.path.join(save_dir, "classification_metrics.txt")
        with open(metrics_save_path, 'w') as f:
            kappa = cohen_kappa_score(test_generator.classes, y_pred)
            auc_val = roc_auc_score(y_true_ohe, y_pred_prob, multi_class='ovr')
            f.write(f"Cohen's Kappa: {kappa}\n")
            f.write(f"AUC: {auc_val}\n")
            f.write(f"Training Time: {train_duration} seconds\n")
            f.write(f"Testing Time: {test_duration} seconds\n")

            cm = confusion_matrix(test_generator.classes, y_pred)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

            report = classification_report(test_generator.classes, y_pred, digits=7, output_dict=True)
            f1_score = report["weighted avg"]["f1-score"]
            precision = report["weighted avg"]["precision"]
            recall = report["weighted avg"]["recall"]
            accuracy = report["accuracy"]

            f.write(f"\n\nPrecision (Weighted Avg): {precision}\n")
            f.write(f"Recall (Weighted Avg): {recall}\n")
            f.write(f"F1-Score (Weighted Avg): {f1_score}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write("\n\nFull Classification Report:\n")
            f.write(classification_report(test_generator.classes, y_pred, digits=7))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

        df_cm = pd.DataFrame(cm, index=np.arange(cm.shape[0]), columns=np.arange(cm.shape[1]))
        plt.figure(figsize=(10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

        history_save_path = os.path.join(save_dir, 'history.txt')
        with open(history_save_path, 'w') as f:
            for key, value_list in history.history.items():
                f.write(f"{key}: {', '.join(map(str, value_list))}\n")
