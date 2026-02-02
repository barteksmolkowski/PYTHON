# My AI Engine 2026 - Science Database

Project of a custom image processing and neural network building library from scratch.

## Project Structure

```text
project/
│
├── common_utils.py          ← (Mózg systemu: funkcja build_all)
├── config.json
├── core.py                  ← (Łącznik: BrainEngine / Facade)
├── main.py                  ← (Punkt startowy aplikacji)
├── pyproject.toml           ← (Zależności i konfiguracja pip/pytest)
├── README.md                ← Opis projektu
├── settings.json            ← (Parametry: ścieżki, learning_rate, size)
├── test_pipeline.py         ← interactive_inspection
│
├── preprocessing/           ← PRZETWARZANIE WSTĘPNE
│   ├── __init__.py
│   ├── augmentation.py      ← DataAugmentation
│   ├── conversion.py        ← ImageToMatrixConverter
│   ├── convolution.py       ← ConvolutionActions
│   ├── decorators.py        ← [...]
│   ├── geometry.py          ← Resize, Padding
│   ├── grayscale.py         ← GrayScaleProcessing
│   ├── io_image.py          ← ImageLoader, DataExporter
│   ├── normalization.py     ← Normalization
│   ├── pipeline.py          ← ImageDataPreprocessing, TransformPipeline
│   ├── pooling.py           ← Pooling
│   └── thresholding.py      ← Thresholding
│
├── features/                ← WYDOBYWANIE CECH (Manualne)
│   ├── __init__.py
│   ├── edges.py             ← (Sobel, Prewitt)
│   ├── extractor.py         ← (Główny FeatureExtractor)
│   └── hog.py               ← (Histogram of Oriented Gradients)
│
├── data/                    ← ZARZĄDZANIE DANYCH
│   ├── __init__.py
│   ├── batch.py             ← (Class: BatchProcessing - pakiety dla sieci)
│   ├── cache.py             ← (Class: CacheManager - zapis .npz)
│   ├── dataset.py           ← (Class: Dataset - dostęp do próbek)
│   ├── downloader.py        ← (Class: DataDownloader - Requests API)
│   └── metadata.json        ← (Baza danych o zdjęciach/etykietach)
│
├── nn/                      ← TWOJA SIEĆ NEURONOWA
│   │
│   ├── layers/              ← MAGAZYN CZĘŚCI ZAMIENNYCH
│   │   ├── __init__.py      ← (Agregator warstw)
│   │   ├── activation.py    ← (Classes: ReLU, Sigmoid, Softmax)
│   │   ├── base.py          ← (LayerProtocol - wzorzec dla wszystkich)
│   │   ├── conv_layer.py    ← (Class: Conv2DLayer)
│   │   ├── dropout.py       ← (Class: DropoutLayer)
│   │   ├── flatten.py       ← (Class: FlattenLayer)
│   │   └── linear.py        ← (Class: LinearLayer / Dense)
│   │
│   ├── __init__.py
│   ├── model.py             ← (Class: NeuralNetwork / Sequential)
│   └── tensor.py            ← (Aliasy: Tensor, Mtx, Shape)
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
    │   ├── __init__.py
    │   ├── test_augmentation.py
    │   ├── test_conversion.py
    │   ├── test_convolution.
    │   ├── test_decorators.py
    │   ├── test_geometry.py   
    │   ├── test_grayscale.py     
    │   ├── test_io_image.py
    │   ├── test_normalization.py
    │   ├── test_pipeline.py
    │   ├── test_pooling.py
    │   └── test_thresholding.py
    │
    ├── test_features/
    │   ├── __init__.py
    │   ├── test_edges.py
    │   ├── test_extractor.py
    │   └── test_hog.py
    │
    ├── test_data/
    │   ├── __init__.py
    │   ├── test_batch.py
    │   ├── test_cache.py    
    │   ├── test_dataset.py
    │   └── test_downloader.py
    │
    ├── test_nn/
    │   │
    │   ├── test_layers/
    │   │   ├── __init__.py
    │   │   ├── test_activation.py
    │   │   ├── test_base.py
    │   │   ├── test_conv_layer.py
    │   │   ├── test_dropout.py
    │   │   ├── test_flatten.py        
    │   │   └── test_linear.py
    │   │
    │   ├── __init__.py
    │   ├── test_model.py
    │   └── test_tensor.py
    │
    └── test_training/
        ├── __init__.py
        ├── test_loss.py
        ├── test_optimizer.py
        └── test_trainer.py
```