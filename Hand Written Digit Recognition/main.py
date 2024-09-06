# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow
import seaborn as sns

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values of the images to be between 0 and 1
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Add an extra dimension to the data to represent the single color channel (grayscale)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Shuffle the training data
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Define the model architecture
model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1), activation='relu'))
model.add(AveragePooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Print a summary of the model architecture
model.summary()

# Compile the model with the Adam optimizer and categorical cross-entropy loss function
model.compile(optimizer='adam', loss=tensorflow.losses.categorical_crossentropy, metrics=['accuracy'])

# Define early stopping and model checkpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5, verbose=1)
mc = ModelCheckpoint("./bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
cb = [es, mc]

his = model.fit(X_train, y_train, epochs=50, validation_split=0.3, callbacks=cb)

# Calculate and print the confusion matrix and F1 score for the test set predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
print('Confusion Matrix:')
print(cm)
print('F1 Score:', f1)

# Plot a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Print the number of correct predictions for each digit class in the test set
for i in range(10):
    print(f"Digit {i}: {cm[i, i]} out of {np.sum(cm[i, :])} predicted correctly")


# Plot the training and validation accuracy over epochs
plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss over epochs
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Calculate and plot the accuracy of correct predictions for each digit class in the test set
correct_predictions = [cm[i, i] for i in range(10)]
total_predictions = [np.sum(cm[i, :]) for i in range(10)]
accuracy = [correct_predictions[i] / total_predictions[i] for i in range(10)]
plt.bar(range(10), accuracy)
plt.xticks(range(10))
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.title('Accuracy of Predictions for Each Digit')
plt.show()

model.save("bestmodel.h5")
model_s = keras.models.load_model("C://Users//abc//PycharmProjects//pythonProject//bestmodel.h5")