from typing import (
    Any,
    Callable,
    Dict,
    List,
    ParamSpec,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

P = ParamSpec("P")
T_Obj = TypeVar("T_Obj")
T_Val = TypeVar("T_Val", int, float, str)
"""
T_Val | Generic Scalar Parameter
---
**Format**: `int | float | str`
**Range**:  Variable (Type-dependent)
**Description**: 
A bound TypeVar used for scalar value extraction and validation. 
Ensures that the returned parameter maintains the type of the provided default value.
**Usage**:
Used in `get_v` to safely extract configuration parameters like `h`, `w`, `angle`, or `fill`.
"""
T: TypeAlias = npt.NDArray[np.float32]
"""
T | Generic Multi-dimensional Tensor
---
**Format**: `[...]` (Variable dimensions)
**Range**:  `[0.0, 1.0]` or Arbitrary (float32)
**Description**: 
A universal array with undefined dimensions for shape-agnostic operations.

**Usage**:
Generic mathematical transformations or abstract tensor operations.
"""

T1D: TypeAlias = npt.NDArray[np.float32]
"""
T1D | Feature Vector
---
**Format**: `[Features]`
**Range**:  `[-inf, inf]` (float32)
**Description**: 
A flattened 1D array representing extracted features or descriptors.

**Usage**:
Input for Linear/Dense layers or distance metric calculations (e.g., HOG).
"""

T2D: TypeAlias = npt.NDArray[np.float32]
"""
T2D | Feature Matrix / Batch
---
**Format**: `[Batch, Features]`
**Range**:  `[-inf, inf]` (float32)
**Description**: 
A 2D array typically representing a batch of feature vectors or a projection.

**Usage**:
Standard input for Multi-Layer Perceptrons (MLP).
"""

T3D: TypeAlias = npt.NDArray[np.float32]
"""
T3D | Sequential Tensor
---
**Format**: `[Batch, Sequence, Hidden]`
**Range**:  `[-inf, inf]` (float32)
**Description**: 
A 3D array used for sequential data, time-series, or block-based feature maps.

**Usage**:
Input for Recurrent Neural Networks (RNN) or Transformers.
"""

T4D: TypeAlias = npt.NDArray[np.float32]
"""
T4D | 4D Batch Tensor (NCHW)
---
**Format**: `[Batch, Channels, Height, Width]`
**Range**:  `[0.0, 1.0]` (float32)
**Description**: 
The standard multi-channel batch format for Deep Learning vision models.

**Usage**:
Direct input for Convolutional Neural Networks (CNNs).
"""

M4D: TypeAlias = npt.NDArray[np.bool_]
"""
M4D | 4D Binary Mask
---
**Format**: `[Batch, Channels, Height, Width]`
**Range**:  `{True, False}` (bool)
**Description**: 
A boolean version of T4D used for spatial filtering or ROI selection.

**Usage**:
Dropout masks, segmentation ground truth, or element-wise logical gating.
"""

ImageGray: TypeAlias = npt.NDArray[Any]
"""
ImageGray | Grayscale Image Matrix
---
**Format**: `[Height, Width]`
**Range**:  `[0, 255]` (uint8) or `[0.0, 1.0]` (float32)
**Description**: 
A 2D matrix representing a single-channel intensity image.

**Usage**:
Input for classic CV algorithms (HOG, SIFT) or single-channel preprocessing.
"""

Padded: TypeAlias = npt.NDArray[np.float32]
"""
Padded | Padded Image Matrix
---
**Format**: `[Height + Pad, Width + Pad]`
**Range**:  `[0.0, 1.0]` (float32)
**Description**: 
A grayscale image with added spatial margins for sliding window or convolution operations.

**Usage**:
Sliding window feature extraction where border context is required.
"""

PaddedImage: TypeAlias = Padded
"""
PaddedImage | Padded Grayscale Image
---
**Format**: `[Height + Pad, Width + Pad]`
**Range**:  `[0.0, 1.0]` (float32)
**Description**: 
An alias for Padded, specifically used for grayscale convolution preprocessing.

**Usage**:
Zero-padding or reflection-padding before applying spatial filters.
"""

ImageRGB: TypeAlias = npt.NDArray[Any]
"""
ImageRGB | RGB Color Image (HWC)
---
**Format**: `[Height, Width, 3]`
**Range**:  `[0, 255]` (uint8) or `[0.0, 1.0]` (float32)
**Description**: 
Standard 3-channel color image in Height-Width-Channel format.

**Usage**:
Input for classic CV operations or visualization before NCHW conversion.
"""

RawImage: TypeAlias = npt.NDArray[np.uint8]
"""
RawImage | Raw Image Buffer (HWC)
---
**Format**: `[Height, Width, Channels]`
**Range**:  `[0, 255]` (uint8)
**Description**: 
Unprocessed image data directly from camera, file, or hardware buffer.

**Usage**:
Initial stage of data loading pipelines before normalization.
"""

LabelsMtx: TypeAlias = npt.NDArray[np.int64]
"""
LabelsMtx | Target Label Vector
---
**Format**: `[Batch]`
**Range**:  `[0, N-1]` (int64)
**Description**: 
A 1D array containing integer class indices for an entire batch.

**Usage**:
Ground truth for Cross-Entropy loss or classification performance metrics.
"""

Label: TypeAlias = int
"""
Label | Class Index
---
**Format**: Scalar
**Range**:  `[0, N-1]` (int)
**Description**: 
A single integer representing a specific object class or category.

**Usage**:
Target value for single sample evaluation or label mapping.
"""

BatchData: TypeAlias = List[T]
"""
BatchData | Processed Tensor Collection
---
**Format**: `List[T]`
**Range**:  Variable (T-dependent)
**Description**: 
A Python list containing processed tensors, typically used before stack operations.

**Usage**:
Intermediate container in custom DataLoader or collation functions.
"""

Sample: TypeAlias = Tuple[T, Label]
"""
Sample | Single Data Point
---
**Format**: `(Tensor, Label)`
**Range**:  N/A (Heterogeneous)
**Description**: 
A standard representation of a single labeled training or inference sample.

**Usage**:
Return type for `__getitem__` in PyTorch-style Dataset classes.
"""

DatasetBatch: TypeAlias = Tuple[BatchData, LabelsMtx]
"""
DatasetBatch | Complete Training Batch
---
**Format**: `(List[T], LabelsMtx)`
**Range**:  N/A (Heterogeneous)
**Description**: 
A fully collated batch of data ready for model ingestion and loss calculation.

**Usage**:
Output of a DataLoader or a custom batch collation function.
"""

JsonDict: TypeAlias = Dict[str, Any]
"""
JsonDict | JSON Object Dictionary
---
**Format**: `Dict[str, Any]`
**Range**:  N/A
**Description**: 
A standard string-keyed dictionary compatible with JSON serialization.

**Usage**:
Configuration files, model state metadata, or API responses.
"""

JsonList: TypeAlias = List[JsonDict]
"""
JsonList | JSON Array
---
**Format**: `List[JsonDict]`
**Range**:  N/A
**Description**: 
A collection of JSON objects, typically representing a dataset of records.

**Usage**:
Batch record export or sequence-based configuration data.
"""

JsonData: TypeAlias = Union[JsonDict, JsonList]
"""
JsonData | Unified JSON Structure
---
**Format**: `Union[Dict, List]`
**Range**:  N/A
**Description**: 
A general type for any valid JSON-serializable data structure.

**Usage**:
Generic data loaders or configuration parsers.
"""

FilePath: TypeAlias = str
"""
FilePath | Filesystem Path String
---
**Format**: `str`
**Range**:  N/A
**Description**: 
A string representing a valid absolute or relative path to a file or directory.

**Usage**:
Input for IO operations, model checkpoints, or dataset roots.
"""

ImageBytes: TypeAlias = bytes
"""
ImageBytes | Encoded Image Buffer
---
**Format**: `bytes`
**Range**:  `0x00 - 0xFF` (Binary)
**Description**: 
Raw encoded image data (e.g., JPEG, PNG, WebP) stored as a byte stream.

**Usage**:
Input for decoding libraries (OpenCV, PIL) or data transmission via APIs.
"""

DataResult: TypeAlias = Union[FilePath, ImageBytes, JsonData]
"""
DataResult | Unified Data Source
---
**Format**: `Union[str, bytes, Dict/List]`
**Range**:  N/A
**Description**: 
A unified union type representing various results of data fetching operations.

**Usage**:
Return type for generic data providers, loaders, or fetcher services.
"""

Shape: TypeAlias = Tuple[int, ...]
"""
Shape | Tensor Dimensions
---
**Format**: `Tuple[int, ...]`
**Range**:  `[0, inf)` (int)
**Description**: 
A tuple of integers representing the size of each dimension in a multidimensional array.

**Usage**:
Input for tensor initialization (e.g., `np.zeros(shape)`) or shape validation.
"""

ClassType: TypeAlias = Type[T_Obj]
"""
ClassType | Type Reference
---
**Format**: `Type[T_Obj]`
**Range**:  N/A
**Description**: 
A reference to a class type itself, rather than a specific instance of that class.

**Usage**:
Used in class decorators, factories, or dependency injection systems.
"""

ProcessorFunc: TypeAlias = Callable[[ImageGray], T]
"""
ProcessorFunc | Image Transformation Signature
---
**Format**: `(ImageGray) -> T`
**Range**:  N/A
**Description**: 
Standard signature for functions that transform a grayscale image into a processed tensor.

**Usage**:
Contract for functional pipelines, feature extractors, or image normalizers.
"""

MetricsDict: TypeAlias = Dict[str, Union[float, int, str]]
"""
MetricsDict | Performance Metadata Store
---
**Format**: `Dict[str, float | int | str]`
**Range**:  N/A
**Description**: 
A key-value store for execution metrics, performance data, or diagnostic metadata.

**Usage**:
Used by decorators to return execution time, status, or accuracy benchmarks.
"""

ResultWithMetrics: TypeAlias = Tuple[T, MetricsDict]
"""
ResultWithMetrics | Composite Processing Result
---
**Format**: `(T, MetricsDict)`
**Range**:  N/A
**Description**: 
A composite return type containing both the processed data and its associated metrics.

**Usage**:
Standard return type for decorated processors to ensure metadata persistence.
"""
"""Declaration "ResultWithMetrics" is obscured by a declaration of the same name"""

Strings: TypeAlias = Union[str, List[str]]
"""
Strings | Unified String Input
---
**Format**: `str | List[str]`
**Range**:  N/A
**Description**: 
A flexible type for text data, accepting either a single string or a collection of strings.

**Usage**:
Configuration keys, labels, or filesystem path collections.
"""

FuncDec: TypeAlias = Union[Callable[P, T], List[Callable[P, T]]]
"""
FuncDec | Functional Decorator or Pipeline
---
**Format**: `Callable[P, T] | List[Callable]`
**Range**:  N/A
**Description**: 
A single callable or a list of callables that preserve the original signature via ParamSpec.

**Usage**:
Pre-processing pipelines or multi-stage functional transformations.
"""

ClsDec: TypeAlias = Callable[[ClassType], ClassType]
"""
ClsDec | Class Decorator Signature
---
**Format**: `(ClassType) -> ClassType`
**Range**:  N/A
**Description**: 
A decorator for class types that ensures the returned object maintains the ClassType identity.

**Usage**:
Registering models, injecting class-level metrics, or extending metadata.
"""
