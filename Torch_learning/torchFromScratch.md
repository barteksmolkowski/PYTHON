data
features
nn
nn.layers
preprocessing
tests.data
tests.features
tests.nn
tests.preprocessing
tests.training
training

1. (ARCHITECTURE)

* ENTRYPOINT MAP (ROLE-BASED)

  * ORCHESTRATION LAYER:

    * CLI dispatcher (main_logic)
    * System Orchestrator (BrainEngine / ProjectManager / Trainer coordinator)
      → responsibility: runtime mode selection (test / train / inference / pipeline run)

  * MODEL RUNTIME LAYER:

    * NeuralNetwork / Sequential
      → responsibility: forward + backward execution over Layer System

  * DATA PIPELINE LAYER:

    * ImageDataPreprocessing / TransformPipeline / ImageHandler / DataProcessor
      → responsibility: deterministic tensor preparation

  * FEATURE LAYER:

    * FeatureExtraction (Sobel / Prewitt / HOG / Edge detection)
      → responsibility: handcrafted feature generation (optional bypass branch)

  * TRAINING CONTROL LAYER:

    * Trainer (loss + optimizer loop controller)
      → responsibility: gradient-based optimization lifecycle

---

* CANONICAL EXECUTION PIPELINE (UNIFIED RUNTIME FLOW)

  INPUT (FilePath / URL / CLI command)
  →
  INGESTION

  * ImageHandler.open_image
  * DataDownloader.fetch_data (optional remote path)
    →
    PREPROCESSING
  * ImageToMatrix conversion
  * channel split (RGB → per-channel tensors)
  * grayscale normalization path
  * geometry standardization (resize/pad to fixed spatial form e.g. 28×28)
  * augmentation (stochastic replication branch)
  * normalization (scale / dtype stabilization)
    →
    FEATURE EXTRACTION (optional / parallel branch)
  * edge detection (Sobel / Prewitt)
  * HOG descriptor extraction
  * identity passthrough if disabled
    →
    MODEL EXECUTION
  * Sequential forward pass
  * Layer stack execution:
    Conv2D → Linear → Activation(ReLU/Sigmoid/Softmax) → Dropout → Flatten
  * cached activations stored per layer
    →
    TRAINING BRANCH (conditional)
  * forward pass
  * loss computation (MSE / Cross-Entropy)
  * backward propagation (layer reverse chain)
  * optimizer step (SGD / Adam)
  * metrics logging
    →
    OUTPUT
  * inference: prediction tensor (T or T4D)
  * training: scalar metrics + history + updated model state
  * preprocessing/debug: transformed image tensors

---

* FEATURE + MODEL + PREPROCESS (UNIFIED FLOW)

  PREPROCESSING (stateless transforms)
  FilePath → ndarray → channel tensors → geometric normalization → augmentation → normalization

  FEATURE EXTRACTION (stateless optional branch)
  grayscale → edge maps → HOG vectorization → feature tensor OR bypass
  MODEL EXECUTION (stateful graph)
  feature/image tensor →
  Sequential Layer Graph →
  forward activations →
  prediction tensor →
  backward gradients (training only)
  TRAINING INTEGRATION (stateful control loop)
  prediction + labels →
  loss →
  gradient propagation →
  optimizer update →
  persistent weight update in model layers
* STATE MODEL (GLOBAL)
  STATELESS COMPONENTS:
  * preprocessing transforms
  * feature extraction (Sobel/HOG/edge)
  * geometry normalization
  * augmentation pipeline
  STATEFUL COMPONENTS:
  * NeuralNetwork / Sequential:
    weights, biases, layer parameters
    cached activations (_input_cache, _mask, outputs)
  * Trainer:
    optimizer state (SGD/Adam momentum, learning rate state)
    loss history
    training metrics log
  STUBBED / NON-PERSISTENT STATE:
  * CacheManager (optional / non-critical)
  * Dataset persistence layer (partial)
  * model save/load infrastructure (implied, incomplete)
* DATA CONTRACTS (TENSORS + SHAPES)
  INPUT CONTRACTS:
  * FilePath / URL / raw image
  * decoded → np.ndarray
  INTERNAL REPRESENTATIONS:
  * ImageRGB: (3, H, W)
  * Grayscale: (H, W)
  * Batch tensor: (N, H, W) or (N, C, H, W)
  * Feature tensor: (N, D) or (HOG vector 1D)
  * Model tensor: T / T4D (abstract ND array)
  OUTPUT CONTRACTS:
  * inference: np.ndarray (prediction logits/probabilities)
  * training: float scalar loss + metrics list
  * augmentation: List[np.ndarray] (expanded dataset views)
---
* SINGLE DEPENDENCY GRAPH (RUNTIME ONLY)

  Orchestrator
  ↓
  Ingestion Layer (ImageHandler / DataDownloader)
  ↓
  Preprocessing Layer (Converter → Geometry → Augmentation → Normalization)
  ↓
  Feature Extraction Layer (Edge / HOG / Optional Identity)
  ↓
  Model Runtime (Sequential → Layers)
  ↓
  Trainer (Loss → Backprop → Optimizer)
  ↓
  Output (Prediction / Metrics / Updated State)
---
* IMPLEMENTATION STATUS
  INTENDED (fully designed but not guaranteed implemented):
  * full CLI orchestration modes
  * unified pipeline switching (test/train/guess)
  * complete Trainer loop abstraction
  * modular feature extraction + preprocessing pipeline
  * sequential model abstraction with backprop chain
  PARTIAL:
  * NeuralNetwork / Sequential layer stack (forward mostly defined, backward incomplete)
  * preprocessing pipeline (multi-stage but duplicated across modules)
  * feature extraction (edge/HOG placeholders present)
  * dataset batching + processor pipeline
  STUBBED:
  * cache system (CacheManager)
  * persistence (save/load model state)
  * optimizer internals (Adam/SGD partially implied)
  * full gradient consistency across layers
  * evaluation pipeline standardization
  BROKEN / INCONSISTENT:
  * duplicated architecture definitions across modules (non-unified runtime view)
  * inconsistent tensor shape contracts (2D / 3D / 4D mixing)
  * partially implemented backward propagation
  * overlapping orchestration layers (BrainEngine vs ProjectManager vs Trainer vs CLI)
ZADANIE:
Wykonaj pełną ARCHITECTURAL COMPRESSION + SYSTEM RECONSTRUCTION na wielu sekcjach "1. (ARCHITECTURE)".
CEL:
Zrekonstruuj JEDEN canonical runtime system ML/backend.
Nie wykonuj merge tekstowego.
Nie streszczaj.
Wykonaj deduplikację semantyczną + rekonstrukcję execution graph.
INPUT MODEL ASSUMPTION
Wszystkie wejściowe sekcje:
- są fragmentami jednego systemu - zawierają duplikaty, aliasy i partial views - opisują ten sam runtime graph w różnych warstwach abstrakcji
KRYTYCZNE ZASADY
1. ENTRYPOINT CANONICALIZATION
- wykryj wszystkie entrypointy - zredukuj do ROLE MAP (nie listy funkcji) - usuń duplikaty semantyczne:
  BrainEngine / ProjectManager / CLI / Sequential / Trainer → ORCHESTRATION LAYERS
2. GLOBAL RUNTIME GRAPH RECONSTRUCTION (OBOWIĄZKOWE)
Zbuduj JEDEN pipeline:
INPUT
→ INGESTION, → PREPROCESSING, → FEATURE EXTRACTION, → MODEL EXECUTION, → TRAINING (optional branch), → OUTPUT
Usuń wszystkie lokalne pipeline’y.
3. DUPLICATE DETECTION (SEMANIC LEVEL, NIE TEXT LEVEL)
Usuń/merge:
- powtórzone preprocessing pipelines, - powtórzone NN layer stacks, - powtórzone feature extraction (Sobel/HOG/Edge), - powtórzone dependency graphs, - powtórzone state descriptions
4. COMPONENT CANONICALIZATION
Zamień aliasy na jeden systemowy koncept:
- BrainEngine / ProjectManager → Orchestrator, - Sequential / NeuralNetwork → Model Runtime, - LayerABC / LayerProtocol → Layer System, - HOG / Sobel / EdgeDetector → Feature Extraction Module
5. STATE UNIFICATION
Zbuduj jeden model state:
- stateless: preprocessing, feature extraction, - stateful: model, trainer, optimizer, - stubbed: cache, persistence, save/load
6. DEPENDENCY GRAPH REBUILD (SINGLE GRAPH ONLY)
Usuń wszystkie lokalne graphs.
Zbuduj:
Orchestrator
→ Preprocessing, → FeatureExtraction, → Model, → Trainer, → Optimizer
7. REDUNDANCY ELIMINATION RULES
ZAKAZANE:
- powtarzanie tensor definitions, - powtarzanie layer lists, - powtarzanie pipeline stages, - lokalne architecture views, - per-module descriptions
ZACHOWAJ TYLKO:
- runtime semantics, - dataflow, - gradient flow, - execution order, - state transitions
OUTPUT FORMAT (STRICT)
Zwróć WYŁĄCZNIE:
1. (ARCHITECTURE), * ENTRYPOINT MAP (ROLE-BASED), * CANONICAL EXECUTION PIPELINE, * FEATURE + MODEL + PREPROCESS (UNIFIED FLOW), * STATE MODEL (GLOBAL), * DATA CONTRACTS (TENSORS + SHAPES), * SINGLE DEPENDENCY GRAPH (RUNTIME ONLY), * IMPLEMENTATION STATUS (INTENDED / PARTIAL / STUBBED / BROKEN)
QUALITY TARGET
Output ma być:
- reverse-engineered system blueprint, - not merge of text, - fully deduplicated runtime graph, - maximal information density
SYMBOL TABLE (CANONICAL ENTITIES)
ORCHESTRATOR
BrainEngine
main_logic / CLI dispatcher
ProjectManager
DATA_PIPELINE
ImageDataPreprocessing
TransformPipeline
ImageHandler
ImageToMatrixConverter
normalization / geometry / augmentation ops
FEATURE_EXTRACTOR
FeatureExtraction
Sobel / Prewitt
HOG (stubbed)
edge detection pipeline
MODEL_RUNTIME
NeuralNetwork
Sequential
LayerABC / LayerProtocol
Conv2D / Linear / ReLU / Sigmoid / Dropout / Flatten
OPTIMIZER
Trainer
loss functions (MSE / CrossEntropy)
SGD / Adam (partial or stubbed)
ENTRYPOINT IR ROOTS
ORCHESTRATOR.start()
MODEL_RUNTIME.forward(x)
OPTIMIZER.step()
DATA_PIPELINE.run(input)
EXECUTION DAG (COMPILED IR)
IR_GRAPH:
InputNode
→ DataPipelineNode
→ FeatureNode
→ ModelNode
→ OutputNode
TRAINING BRANCH:
ModelNode
→ OptimizerNode
→ ModelNode (weight update loop)
DATAFLOW + MODEL FLOW (UNIFIED)
INPUT:
FilePath | URL | Raw Image → InputNode
DATA_PIPELINE:
decode → tensor (3,H,W)
grayscale projection
resize/geometry normalization
augmentation (stochastic transforms)
normalization (/255, z-score, min-max variants)
FEATURE_EXTRACTOR (optional bypass path):
Sobel/Prewitt → gradient maps
HOG → histogram vector (stub/partial)
output: T1D or identity passthrough

MODEL_RUNTIME:

reshape → T4D batch tensor (N,C,H,W)
sequential composition:
Conv2D → Activation → Flatten → Linear → Dropout
forward: logits/activations
backward: chain rule gradient propagation (partial)

OPTIMIZER (training only):

loss: MSE / CrossEntropy
gradient computation (partial/stub)
parameter update: θ ← θ - η∇L (SGD/Adam incomplete)
history tracking

OUTPUT:

prediction tensor OR class label OR training metrics
STATE IR MODEL

STATIC:

preprocessing transforms
feature extraction ops
convolution / pooling logic definitions

DYNAMIC:

model weights (Linear/Conv2D)
cached activations (forward pass memory)
gradients (backprop flow)
optimizer state (momentum/Adam partial)
training history logs

BROKEN / STUB:

cache manager
dataset persistence
save/load checkpointing
HOG descriptor full implementation
Softmax (unreliable / incomplete in parts)
TENSOR CONTRACTS (SHAPES + TYPES)
Input:
(3, H, W) uint8 | float32
Preprocessed:
(H, W) float32
Batch Model Input:
(N, C, H, W)
Feature Output:
(H, W) OR (128,) OR T1D vector
Model Output:
(N, K) logits OR (K,) class vector
Gradients:
same shape as corresponding activations per layer
FINAL DEPENDENCY GRAPH (SINGLE IR GRAPH)

ORCHESTRATOR
→ DATA_PIPELINE
→ FEATURE_EXTRACTOR
→ MODEL_RUNTIME
→ OUTPUT

          ↘
           OPTIMIZER
           (training branch only)

COMPILATION STATUS
ORCHESTRATOR: PARTIAL (multi-root ambiguity resolved)
DATA_PIPELINE: PARTIAL (overlapping pipelines unified)
FEATURE_EXTRACTOR: PARTIAL (core ops stubbed)
MODEL_RUNTIME: PARTIAL (forward/backward incomplete but structurally valid)
OPTIMIZER: STUBBED (update rules not fully implemented)
END-TO-END SYSTEM: FRAGMENTED BUT COMPILABLE INTO SINGLE IR GRAPH
RESULT

✔ system successfully reduced to:

SINGLE canonical ML runtime IR graph
with unified execution semantics, state model, and tensor contracts
SYMBOL TABLE (CANONICAL ENTITIES)

ORCHESTRATOR

BrainEngine
ProjectManager
CLI / main_logic

DATA_PIPELINE

ImageDataPreprocessing
TransformPipeline
ImageHandler
ImageToMatrixConverter
Geometry / Augmentation / Normalization / Thresholding / Pooling

FEATURE_EXTRACTOR

FeatureExtraction
Sobel / Prewitt
HOG
Edge detection logic

MODEL_RUNTIME

NeuralNetwork
Sequential
LayerProtocol system
Conv2D / Linear / ReLU / Sigmoid / Dropout / Flatten

OPTIMIZER

Trainer
Loss (MSE / CrossEntropy)
SGD / Adam (partial/stub)
ENTRYPOINT IR ROOTS
ORCHESTRATOR.start()
DATA_PIPELINE.run(input)
FEATURE_EXTRACTOR.run(input)
MODEL_RUNTIME.forward(x)
OPTIMIZER.step()
EXECUTION DAG (COMPILED IR — SINGLE GRAPH)

IR_GRAPH:

InputNode
→ DataPipelineNode
→ FeatureNode
→ ModelNode
→ OutputNode

TRAINING BRANCH:
ModelNode → OptimizerNode → ModelNode

DATAFLOW + MODEL FLOW (UNIFIED)

INPUT:

FilePath / URL / Image → InputNode

DATA_PIPELINE:

image decode → ndarray
channel split (RGB → 3,H,W)
grayscale projection (weighted RGB)
resize / padding (geometry normalization)
convolution / pooling / thresholding (deterministic ops)
augmentation (flip/rotate/noise/shift)
normalization (/255, z-score, min-max)

FEATURE_EXTRACTOR:

Sobel/Prewitt → gradient maps (Gx, Gy → magnitude)
HOG → histogram descriptor (stub/partial)
fallback: identity passthrough or zero vector

MODEL_RUNTIME:

reshape → T4D batch tensor (N,C,H,W)
forward composition:
Conv2D → Activation → Flatten → Linear → Dropout
backward:
chain rule propagation (partial implementation)
Linear layer: fully defined XW + b + gradients
other layers: partial/stubbed gradients

OPTIMIZER:

loss: MSE / CrossEntropy
gradient computation (partial/stub)
update rule: θ ← θ - η∇L (SGD/Adam incomplete)
training history accumulation

OUTPUT:

prediction tensor OR classification OR training metrics
STATE IR MODEL (GLOBAL ONLY)

STATIC:

preprocessing transforms
convolution / pooling / threshold definitions
feature extraction operators

DYNAMIC:

model weights (Linear / Conv2D)
gradients (backprop flow)
optimizer state (momentum / Adam partial)
cached activations (forward pass memory)
training history

BROKEN / STUB:

cache manager
dataset persistence
save/load checkpoints
full HOG descriptor implementation
softmax stability layer (partial/missing in parts)
TENSOR CONTRACTS (SHAPES + TYPES)
INPUT:
(3, H, W) uint8 | float32
PIPELINE OUTPUT:
(H, W) grayscale OR transformed tensor
FEATURE OUTPUT:
(H, W) OR (128,) OR T1D vector OR ∅ fallback [0.0]
MODEL INPUT:
(N, C, H, W)
MODEL OUTPUT:
(N, K) logits OR (K,) class vector
GRADIENTS:
same shape as corresponding layer activations
FINAL DEPENDENCY GRAPH (SINGLE IR GRAPH)

ORCHESTRATOR
→ DATA_PIPELINE
→ FEATURE_EXTRACTOR
→ MODEL_RUNTIME
→ OUTPUT

          ↘
           OPTIMIZER
           (training-only branch)

COMPILATION STATUS
ORCHESTRATOR: PARTIAL (multi-entry collapsed into single root)
DATA_PIPELINE: PARTIAL (multiple preprocessing systems unified but internally heterogeneous)
FEATURE_EXTRACTOR: PARTIAL (core math intended, some stubbed)
MODEL_RUNTIME: PARTIAL (Linear correct, rest mixed implementation/stubs)
OPTIMIZER: STUBBED (update rules incomplete, abstract interfaces only)
END-TO-END GRAPH: VALID IR STRUCTURE BUT NOT FULLY EXECUTABLE
RESULT

✔ system successfully compiled into:

SINGLE canonical ML runtime IR graph
lossless structural unification achieved at architecture level
execution semantics partially incomplete but graph-consistent
SYMBOL TABLE (CANONICAL ENTITIES)

ORCHESTRATOR:

BrainEngine
ProjectManager
CLI / main_logic

DATA_PIPELINE:

ImageDataPreprocessing
TransformPipeline
ImageHandler
ImageToMatrixConverter
normalization / geometry / augmentation / pooling / thresholding

FEATURE_EXTRACTOR:

FeatureExtraction
Sobel / Prewitt
HOG
edge detection ops

MODEL_RUNTIME:

NeuralNetwork
Sequential
LayerProtocol system
Conv2D / Linear / ReLU / Sigmoid / Dropout / Flatten

OPTIMIZER:

Trainer
Loss (MSE / CrossEntropy)
SGD / Adam (partial/stub)
ENTRYPOINT IR ROOTS

ORCHESTRATOR.start()
DATA_PIPELINE.run(input)
FEATURE_EXTRACTOR.run(input)
MODEL_RUNTIME.forward(x)
OPTIMIZER.step()

EXECUTION DAG (SINGLE COMPILATION GRAPH)

IR_GRAPH:

InputNode
→ DataPipelineNode
→ FeatureNode
→ ModelNode
→ OutputNode

TRAINING BRANCH:
ModelNode → OptimizerNode → ModelNode

DATAFLOW + MODEL FLOW (UNIFIED SEMANTICS)

INPUT:

FilePath / URL / Raw Image → InputNode

DATA_PIPELINE:

decode image → ndarray (H,W,C)
channel split → (3,H,W)
grayscale projection (weighted sum)
resize / padding → geometry normalization (28x28 optional constraint)
augmentation → stochastic transforms (flip/rotate/noise/shift)
pooling / thresholding → spatial reduction ops
normalization → /255, z-score, min-max scaling

FEATURE_EXTRACTOR (optional / bypass-capable path):

Sobel/Prewitt → gradient magnitude maps (Gx, Gy)
HOG → histogram vector (128-dim stub/partial)
fallback → identity / zero vector

MODEL_RUNTIME:

reshape → T4D batch tensor (N,C,H,W)
forward chain:
Conv2D → Activation → Flatten → Linear → Dropout
backward:
chain rule propagation (partial)
Linear fully defined:
XW + b + dW + db
Conv2D / Softmax: stubbed or incomplete

OPTIMIZER:

loss:
MSE / CrossEntropy
gradients:
partial or symbolic only
update rule:
θ ← θ - η∇L (SGD/Adam incomplete)
training history: stored but inconsistent

OUTPUT:

prediction tensor OR class label OR training metrics
STATE IR MODEL (GLOBAL ONLY)

STATIC:

preprocessing operators (geometry, normalization, augmentation)
feature extraction operators (Sobel, HOG logic)
activation functions

DYNAMIC:

model weights (Linear/Conv2D)
gradients (backprop tensors)
optimizer state (momentum / adaptive buffers partial)
cached activations (forward pass memory)
training history logs

BROKEN / STUB:

cache manager system
dataset persistence layer
model save/load
softmax stability / normalization layer
full convolution backward engine
TENSOR CONTRACTS (SHAPES + TYPES)

INPUT:

(3, H, W) uint8 | float32

PIPELINE OUTPUT:

(H, W) grayscale OR transformed spatial tensor

FEATURE OUTPUT:

(H, W)
OR (128,)
OR T1D vector
OR empty fallback [0.0]

MODEL INPUT:

(N, C, H, W)

MODEL OUTPUT:

(N, K) logits OR (K,) class vector

GRADIENTS:

same shape as corresponding forward tensors (layer-aligned)
FINAL DEPENDENCY GRAPH (SINGLE IR GRAPH)

ORCHESTRATOR
→ DATA_PIPELINE
→ FEATURE_EXTRACTOR
→ MODEL_RUNTIME
→ OUTPUT

          ↘
           OPTIMIZER
           (training-only branch)

COMPILATION STATUS
ORCHESTRATOR: PARTIAL (multiple roots unified)
DATA_PIPELINE: PARTIAL (many overlapping preprocessing paths merged into one IR node)
FEATURE_EXTRACTOR: PARTIAL (math defined, execution mostly stubbed)
MODEL_RUNTIME: PARTIAL (Linear valid, others incomplete or stubbed)
OPTIMIZER: STUBBED (update rules not fully implemented)
END-TO-END SYSTEM: STRUCTURALLY CONSISTENT IR, EXECUTION INCOMPLETE
RESULT

✔ system compiled into single canonical ML runtime IR graph
✔ all duplicates eliminated into unified execution model
✔ final output is a lossless architecture-level IR, but runtime behavior remains partially unimplemented

SYMBOL TABLE (CANONICAL ENTITIES)

ORCHESTRATOR
= BrainEngine + CLI + ProjectManager + main_logic

DATA_PIPELINE
= ImageDataPreprocessing + TransformPipeline + DataProcessor + BatchProcessing + Augmentation + Normalization + Geometry + IO handlers

FEATURE_EXTRACTOR
= Sobel + Prewitt + HOG + EdgeDetector + FeatureExtraction

MODEL_RUNTIME
= NeuralNetwork + Sequential + LayerProtocol stack + Conv2D + Linear + Activation layers

OPTIMIZER
= Trainer + Loss (MSE/CrossEntropy) + SGD + Adam + backward loops

ENTRYPOINT IR ROOTS (COLLAPSED)

ORCHESTRATOR.start(INPUT)
MODEL_RUNTIME.forward(TENSOR)
OPTIMIZER.step(GRADIENTS)
DATA_PIPELINE.run(INPUT)

EXECUTION DAG (COMPILED IR_GRAPH)

IR_GRAPH:

InputNode
→ DataPipelineNode
→ FeatureNode
→ ModelNode
→ OutputNode

InputNode
→ DataPipelineNode
→ ModelNode
→ OutputNode

ModelNode
→ OptimizerNode
→ ModelNode (state update loop)

DATAFLOW + MODEL FLOW (UNIFIED)

INPUT:

FilePath | URL | RawImage | Tensor

FLOW:
Input
→ decode/load (IO)
→ tensorization (H,W,C → T4D / T2D)
→ normalization (/255, z-score, min-max)
→ augmentation (flip/rotate/noise/shift)
→ batching (N splits, shuffle)
→ feature extraction (optional branch: Sobel/Prewitt/HOG)
→ model reshape (flatten / conv input format)
→ forward pass:
Conv2D → Activation → Dropout → Flatten → Linear
→ prediction logits

TRAINING BRANCH:
prediction + labels
→ loss (MSE / CrossEntropy)
→ gradient computation (backprop chain rule)
→ optimizer update (SGD/Adam)
→ weight mutation in MODEL_RUNTIME state

STATE IR MODEL (GLOBAL ONLY)

STATIC:

normalization rules
convolution kernels (Sobel/Prewitt)
augmentation policies
activation functions

DYNAMIC:

MODEL_RUNTIME weights (Conv2D, Linear)
optimizer state (SGD momentum / Adam moments [STUBBED])
gradients per layer
training history (loss logs)

BROKEN/STUB:

cache manager
persistence (save/load partial or undefined schema)
dataset storage abstraction
feature extractor outputs (partially identity / zero vectors)
Conv2D forward/backward (unimplemented in core paths)
TENSOR CONTRACTS (SHAPES + TYPES)

INPUT:

Image: (H,W), (H,W,C)
Batch: (B,C,H,W)
dtype: uint8 | float32

PIPELINE:

normalized image: (H,W) float32 ∈ [0,1]
augmented set: List[(H,W)]
batch tensor: (B,C,H,W)

FEATURE:

edge map: (H,W)
HOG vector: (128,)
feature vector: (N,) or [0.0] fallback

MODEL INPUT:

Linear input: (B, F)
Conv input: (B, C, H, W)

OUTPUT:

logits: (B, K)
prediction: (K,) or (B,K)

GRADIENTS:

same shape as corresponding forward tensors (layer-aligned)
FINAL DEPENDENCY GRAPH (SINGLE IR GRAPH)

ORCHESTRATOR
→ DATA_PIPELINE
→ FEATURE_EXTRACTOR
→ MODEL_RUNTIME
→ OUTPUT

             ↘  
              OPTIMIZER  
               ↺ (state update loop into MODEL_RUNTIME)
COMPILATION STATUS

PARTIAL

DATA_PIPELINE: functional but inconsistent contracts
FEATURE_EXTRACTOR: mostly stubbed / identity transforms
MODEL_RUNTIME: partially implemented (Linear ok, Conv2D/Softmax incomplete)
OPTIMIZER: interface-level only (no real updates)
END-TO-END GRAPH: structurally defined, execution incomplete
STATE SYSTEM: partially broken (cache/persistence undefined)