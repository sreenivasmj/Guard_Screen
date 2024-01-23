from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Define data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'archive/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['children', 'adult']
)

test_generator = test_datagen.flow_from_directory(
    'archive/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['children', 'adult']
)

# Load MobileNetV2 with pre-trained weights (without the top layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)

# Create a smaller model based on MobileNetV2
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with data augmentation and early stopping
model.fit(train_generator, validation_data=test_generator, epochs=50, callbacks=[early_stopping])

# Save the trained model to a file
model.save('trained_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Predict the labels for the test set
y_pred_probs = model.predict(test_generator)
threshold = 0.5  # Adjust the threshold as needed
y_pred_binary = (y_pred_probs > threshold).astype(int)

# Get true labels
y_true = test_generator.classes

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred_binary)
class_report = classification_report(y_true, y_pred_binary)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)
