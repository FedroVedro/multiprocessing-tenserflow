import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist, fashion_mnist
import multiprocessing
import threading
import time

# Функция для создания модели
def create_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Функция для обучения модели
def train_model(dataset_name):
    # Загрузка датасета
    if dataset_name == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "FashionMNIST":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        print(f"Неизвестный датасет: {dataset_name}")
        return

    x_train, x_test = x_train / 255.0, x_test / 255.0  # Нормализация

    # Создание модели
    model = create_model(input_shape=x_train.shape[1:], num_classes=10)

    # Функция мониторинга с использованием threading
    def monitor_training():
        while True:
            print(f"Обучение модели на датасете {dataset_name}...")
            time.sleep(10)

    # Запускаем мониторинг в отдельном потоке
    monitor_thread = threading.Thread(target=monitor_training, daemon=True)
    monitor_thread.start()

    # Обучение модели
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
    print(f"Обучение на датасете {dataset_name} завершено!")

# Главная функция для запуска параллельных процессов
def main():
    datasets = ["MNIST", "FashionMNIST"]
    processes = []

    for dataset in datasets:
        process = multiprocessing.Process(target=train_model, args=(dataset,))
        processes.append(process)
        process.start()

    # Ожидание завершения всех процессов
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
