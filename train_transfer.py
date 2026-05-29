import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# ── Config ────────────────────────────────────────────────────
IMG_SIZE   = 224   # MobileNetV2 requires 224x224
BATCH_SIZE = 32
EPOCHS_1   = 10    # Phase 1: feature extraction
EPOCHS_2   = 10    # Phase 2: fine-tuning

# ── Data ──────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ── Model: MobileNetV2 base ───────────────────────────────────
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Phase 1: freeze all

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── Phase 1: Feature Extraction ───────────────────────────────
print("\n=== Phase 1: Feature Extraction (base model frozen) ===")
print(f"Trainable params: {model.count_params():,}\n")

history1 = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS_1,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('model_phase1.keras', save_best_only=True, verbose=1)
    ]
)

# ── Phase 2: Fine-Tuning ──────────────────────────────────────
base_model.trainable = True
for layer in base_model.layers[:-30]:   # Freeze all but top 30 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Low LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n=== Phase 2: Fine-Tuning (top 30 layers unfrozen) ===")
print(f"Trainable params: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}\n")

history2 = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS_2,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('model_best.keras', save_best_only=True, verbose=1)
    ]
)

# ── Evaluation ────────────────────────────────────────────────
test_set.reset()
loss, accuracy = model.evaluate(test_set, verbose=1)
print(f"\n📊 Test Accuracy : {accuracy*100:.2f}%")
print(f"📊 Test Loss     : {loss:.4f}")

model.save('model_final.keras')
print("\n✅ Saved: model_final.keras")

# ── Single Prediction ─────────────────────────────────────────
from tensorflow.keras.preprocessing import image

def predict(img_path: str) -> str:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x   = image.img_to_array(img) / 255.0
    x   = np.expand_dims(x, axis=0)
    prob = model.predict(x, verbose=0)[0][0]
    label = "dog" if prob > 0.5 else "cat"
    print(f"Prediction: {label} (confidence: {max(prob, 1-prob)*100:.1f}%)")
    return label

predict('dataset/single_prediction/cat_or_dog_1.jpg')