# multiprocessing-tenserflow

Цель проекта в паралельном обучении простых нейронных сетей, используя разные датасеты TensorFlow (MNIST и Fashion MNIST).

Нужные библиотеки :

TensorFlow для построения и обучения нейронных сетей.

multiprocessing для параллельного выполнения процессов.

threading для многопоточности внутри одного из процессов (для мониторинга выполнения обучения).

План проекта:

Главный процесс управляет запуском задач.
Параллельные процессы обучают отдельные модели на разных датасетах.
В одном из процессов будем использовать threading для вывода текущего состояния обучения.

