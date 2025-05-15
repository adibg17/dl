import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load and preprocess CIFAR-10 (as a simulated disaster dataset)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train / 255.0, [224, 224])
x_test = tf.image.resize(x_test / 255.0, [224, 224])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Model architecture using EfficientNetB0
def build_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

# 3. Train model with given optimizer
def train_model(optimizer, label):
    print(f"\nTraining with {label}...")
    model = build_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=1)
    return history

# 4. Define optimizers
optimizers = {
    "SGD": SGD(learning_rate=0.001),
    "Adam": Adam(learning_rate=0.001),
    "RMSprop": RMSprop(learning_rate=0.001)
}

# 5. Train and store histories
histories = {}
for name, opt in optimizers.items():
    histories[name] = train_model(opt, name)

# 6. Plot metrics
def plot_metric(histories, metric):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist.history[metric], label=f'{name} {metric}')
    plt.title(f'{metric.capitalize()} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.show()

plot_metric(histories, 'accuracy')
plot_metric(histories, 'val_accuracy')
plot_metric(histories, 'loss')
plot_metric(histories, 'val_loss')
