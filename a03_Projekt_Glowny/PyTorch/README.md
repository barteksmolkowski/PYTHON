# My AI Engine 2026 - Science Database

Project of a custom image processing and neural network building library from scratch.

## Project Structure

```text
project/
│
├── common_utils.py          ← (Mózg systemu: funkcja build_all)
├── core.py                  ← (Łącznik: BrainEngine / Facade)
├── main.py                  ← (Punkt startowy aplikacji)
├── pyproject.toml           ← (Zależności i konfiguracja pip/pytest)
├── settings.json            ← (Parametry: ścieżki, learning_rate, size)
│
├── preprocessing/           ← PRZETWARZANIE WSTĘPNE
│   ├── __init__.py
│   ├── io.py                ← ImageLoader, DataExporter
│   ├── conversion.py        ← ImageToMatrixConverter
│   ├── geometry.py          ← Resize, Padding, MatrixCreator
│   ├── normalization.py     ← Normalization
│   ├── grayscale.py         ← GrayScaleProcessing
│   ├── thresholding.py      ← Thresholding
│   ├── augmentation.py      ← DataAugmentation
│   ├── convolution.py       ← ConvolutionActions
│   ├── pooling.py           ← Pooling
│   └── pipeline.py          ← ImageDataPreprocessing, TransformPipeline
│
├── features/                ← WYDOBYWANIE CECH (Manualne)
│   ├── __init__.py
│   ├── edges.py             ← (Sobel, Prewitt)
│   ├── hog.py               ← (Histogram of Oriented Gradients)
│   └── extractor.py         ← (Główny FeatureExtractor)
│
├── data/                    ← ZARZĄDZANIE DANYCH
│   ├── __init__.py
│   ├── dataset.py           ← (Class: Dataset - dostęp do próbek)
│   ├── batch.py             ← (Class: BatchProcessing - pakiety dla sieci)
│   ├── cache.py             ← (Class: CacheManager - zapis .npz)
│   ├── downloader.py        ← (Class: DataDownloader - Requests API)
│   └── metadata.json        ← (Baza danych o zdjęciach/etykietach)
│
├── nn/                      ← TWOJA SIEĆ NEURONOWA
│   ├── __init__.py
│   ├── tensor.py            ← (Aliasy: Tensor, Mtx, Shape)
│   ├── model.py             ← (Class: NeuralNetwork / Sequential)
│   └── layers/              ← MAGAZYN CZĘŚCI ZAMIENNYCH
│       ├── __init__.py      ← (Agregator warstw)
│       ├── base.py          ← (LayerProtocol - wzorzec dla wszystkich)
│       ├── linear.py        ← (Class: LinearLayer / Dense)
│       ├── activation.py    ← (Classes: ReLU, Sigmoid, Softmax)
│       ├── dropout.py       ← (Class: DropoutLayer)
│       ├── flatten.py       ← (Class: FlattenLayer)
│       └── conv_layer.py    ← (Class: Conv2DLayer)
│
├── training/                ← PROCES UCZENIA
│   ├── __init__.py
│   ├── loss.py              ← (Classes: MSE, CrossEntropy)
│   ├── optimizer.py         ← (Classes: SGD, Adam)
│   └── trainer.py           ← (Class: Trainer - pętla ucząca)
│
└── tests/                   ← TWOJE 20+ TESTÓW (Lustrzane odbicie)
    ├── __init__.py
    ├── conftest.py          ← (Wspólne "fixture" - np. generowanie testowej Mtx)
    │
    ├── test_preprocessing/
    │   ├── test_io.py
    │   ├── test_conversion.py
    │   ├── test_geometry.py
    │   ├── test_normalization.py
    │   ├── test_grayscale.py
    │   ├── test_thresholding.py
    │   ├── test_augmentation.py
    │   ├── test_convolution.py
    │   ├── test_pooling.py
    │   └── test_pipeline.py
    │
    ├── test_features/
    │   ├── test_edges.py
    │   ├── test_hog.py
    │   └── test_extractor.py
    │
    ├── test_data/
    │   ├── test_dataset.py
    │   ├── test_batch.py
    │   ├── test_cache.py
    │   ├── test_downloader.py
    │   └── test_metadata.json
    │
    ├── test_nn/
    │   ├── test_tensor.py
    │   ├── test_model.py
    │   └── test_layers/
    │       ├── test_base.py
    │       ├── test_linear.py
    │       ├── test_activation.py
    │       ├── test_dropout.py
    │       ├── test_flatten.py
    │       └── test_conv_layer.py
    │
    └── test_training/
        ├── test_loss.py
        ├── test_optimizer.py
        └── test_trainer.py
```