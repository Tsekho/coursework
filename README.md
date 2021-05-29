# Курсовая работа

## Локальное обучение

### Конфигурация

#### Система

- AMD Ryzen 9 5900HX CPU
- NVIDIA GeForce RTX 3070 Laptop GPU

#### ПО

- Windows 10 Home Edition
- CUDA 11.1
- Python 3.8.7 64bit
- PyTorch 1.8.1

#### Параметры

- Набор данных [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - 10 классов по 5000 тренировочных и 1000 валидационных изображений 32✕32px
- 50 эпох
- 64 размер батча
- Сокращение веса 5e-4
- Момент 0.7
- Архитектуры:
  - [SimpleNetV1]("https://arxiv.org/abs/1608.06037")
  - [ResNet18, ResNet34, ResNet50](https://arxiv.org/abs/1512.03385v1)

#### Использование скрипта

- Обучение:
`mpirun -np <n> --hostfile <filename> python3 train.py <args>`
  - `<n>` - количество процессов
  - `<filename>` - хост-файл
  - Аргументы скрипта:
    - Общее:
      - `--network=simplenetv1` - архитектура
        - также принимает `resnet18`, `resnet34`, `resnet50`
      - `--epochs=50` - количество эпох
      - `--warmup_epochs=5` - количество разогревочных эпох с пониженной компрессией
      - `-p=1.0` - доля датасета
      - `--batch_size=64` - размер батча
    - CUDA:
      - `[--cuda_server]` - использовать CUDA с главным процессом
      - `[--cuda_clients]` - использовать CUDA с работниками
    - Параметры:
      - `--learning_rate=0.05` - темп обучения
      - `--milestones=[30, 40]` - перечисление порогов для снижения темпа обучения
      - `--gamma=0.1` - коэффициент снижения темпа обучения
        - `LR = LR * G if LR in MS else LR`
      - `--compression_rate=0.01` - базовый коэффициент компрессии
      - `--warmup_compression_rate=0.5` - разогревочный коэффициент компрессии
        - `CR = min(cr, wcr ** (e + 1)) if E <= W else cr`
      - `-clip_gradient_norm=1` - пороговая норма Фробениуса для каждого тензора в градиенте
      - `--momentum=0.7` - момент накапливаемого локально градиента
      - `[--nesterov]` - использовать момент Нестерова
      - `--weight_decay=5e-4` - ограничение весов
    - Вывод:
      - `[--silent]` - не использовать стандартный вывод
      - `[--tensorboard]` - вести логи для TensorBoard
        - `tensorboard --logdir="runs" --port=6006 --host="localhost"`
      - `[--csv]` - вести CSV логи
      - `--checkpoints=0` - интервал сохранения модели

## Репозитории

- [Deep Gradient Compression](https://github.com/synxlin/deep-gradient-compression) - распределённое обучение с Horovod и PyTorch, имплементация опорного метода [Y.Lin et al. [2017]](2)
  - [GRAdient ComprEssion for distributed deep learning](https://github.com/sands-lab/grace) - основа для авторской имплементации
- [DGS PyTorch](https://github.com/yanring/DGS) - код работы [Z.Yan et al. [2019]](https://dl.acm.org/doi/10.1145/3404397.3404401), реализация других подходов, включая [A.Aji, K.Heafield [2017]](1)

[1]: https://arxiv.org/abs/1704.05021 "A.Aji, K.Heafield [2017]"
[2]: https://arxiv.org/abs/1712.01887 "Y.Lin et al. [2017]"
