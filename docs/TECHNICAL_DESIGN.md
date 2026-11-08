# Quantum ASR – Technical Design Document

**Live Hindi ↔ English speech translation with a hybrid quantum–classical spoken-language identifier**

| | |
|---|---|
| Version | 1.0 (7 Nov 2026) |
| Code | `qasr/` (Python 3.10–3.12, PyTorch 2.6, C++17, CUDA C via NVRTC) |
| Training / inference hardware | HP Victus · Intel Core i5 H-series · NVIDIA RTX 4070 Laptop GPU (8 GB VRAM, Ada, compute capability 8.9) |
| Development hardware | Intel i7-1165G7, 16 GB RAM, no NVIDIA GPU (CPU path + C++ kernel validated here) |
| Operator guide | [`INSTRUCTIONS.md`](../INSTRUCTIONS.md) |

---

## Contents

1. [Goals and requirements](#1-goals-and-requirements)
2. [System architecture](#2-system-architecture)
3. [Why the quantum model handles language detection](#3-why-the-quantum-model-handles-language-detection)
4. [Quantum model design](#4-quantum-model-design)
5. [Low-level simulation engine (C++ / CUDA)](#5-low-level-simulation-engine-c--cuda)
6. [Data plan (AI4Bharat)](#6-data-plan-ai4bharat)
7. [Feature pipeline](#7-feature-pipeline)
8. [Training and evaluation protocol](#8-training-and-evaluation-protocol)
9. [Live inference pipeline](#9-live-inference-pipeline)
10. [Latency engineering](#10-latency-engineering)
11. [Software stack and versions](#11-software-stack-and-versions)
12. [Project structure](#12-project-structure)
13. [Configuration reference](#13-configuration-reference)
14. [Verification and testing](#14-verification-and-testing)
15. [Risks, limitations and mitigations](#15-risks-limitations-and-mitigations)
16. [Implementation plan and future work](#16-implementation-plan-and-future-work)

---

## 1. Goals and requirements

### 1.1 Functional

| ID | Requirement |
|---|---|
| F1 | Capture live microphone audio continuously (16 kHz mono). |
| F2 | Segment speech into utterances automatically (no push-to-talk). |
| F3 | **Detect the spoken language automatically** (Hindi or English) per utterance. |
| F4 | Transcribe Hindi with an AI4Bharat ASR model and English with a Whisper-family model. |
| F5 | Translate Hindi → English and English → Hindi with AI4Bharat IndicTrans2. |
| F6 | Show the source text and translation, optionally speak the translation aloud (TTS). |
| F7 | The language identifier (LID) contains a **trainable variational quantum circuit**, trained and run entirely on a **local simulator**. |
| F8 | Training is reproducible from public AI4Bharat datasets with **one command**. |
| F9 | Export the trained circuit as **OpenQASM 3** so it can run on real quantum hardware (e.g. IBM Quantum via Qiskit). |

### 1.2 Non-functional

| ID | Requirement | Target |
|---|---|---|
| N1 | End-of-speech → translation shown (perceived latency) | **≤ 1.0 s** (goal ≈ 0.6 s) |
| N2 | Compute latency per utterance (LID + ASR + MT) | ≤ 400 ms on RTX 4070 |
| N3 | Quantum LID head latency (batch 1) | ≤ 2 ms on GPU |
| N4 | LID accuracy (in-domain test) | ≥ 97 % |
| N5 | Fits in 8 GB VRAM with all models loaded | yes (≈ 5–6 GB estimated) |
| N6 | Runs offline after models are downloaded | yes |
| N7 | Training interruptible and resumable | yes (every stage caches its output) |
| N8 | Fair scientific comparison | parameter-matched classical baseline trained on identical data |

---

## 2. System architecture

### 2.1 Live inference (data flow)

```
 ┌────────────┐ 32 ms frames ┌──────────────┐ utterance ┌───────────────────────────────────────────┐
 │ Microphone │─────────────▶│ Silero VAD   │──────────▶│ Language identification (hybrid quantum)  │
 │ 16 kHz mono│ sounddevice  │ endpointing  │  (≤ 8 s)  │ Whisper-base encoder → mean/std pooling   │
 └────────────┘              │ pre-roll 300 │           │ → Linear(1024→10) → π·tanh                │
                             │ silence 350ms│           │ → 10-qubit circuit (CUDA kernel) → <Z>    │
                             └──────────────┘           │ → Linear(10→2) → P(hi), P(en)             │
                                                        └──────────────┬────────────────────────────┘
                                         confidence ≥ 0.65 → use it,   │ else reuse previous language
                                     ┌─────────────────────────────────┴───────────────┐
                                     ▼ hi                                              ▼ en
                     ┌──────────────────────────────┐                 ┌──────────────────────────────┐
                     │ IndicConformer-600M (AI4B)   │                 │ distil-whisper large-v3      │
                     │ ONNX Runtime GPU, CTC decode │                 │ fp16, greedy, language="en"  │
                     └──────────────┬───────────────┘                 └──────────────┬───────────────┘
                                    ▼                                                ▼
                     ┌──────────────────────────────┐                 ┌──────────────────────────────┐
                     │ IndicTrans2 indic→en 200M    │                 │ IndicTrans2 en→indic 200M    │
                     │ fp16, greedy                 │                 │ fp16, greedy                 │
                     └──────────────┬───────────────┘                 └──────────────┬───────────────┘
                                    └──────────────────────┬─────────────────────────┘
                                                           ▼
                                   console (colour) · session log (JSONL) · optional TTS
```

### 2.2 Training (offline)

```
 Hugging Face (gated, streamed)          local disk                          GPU
 ┌──────────────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────────────┐
 │ Kathbath  IndicVoices        │   │ data/clips/*.flac   │   │ frozen Whisper-base encoder     │
 │ Shrutilipi Lahaja (Hindi)    │──▶│ data/manifest.jsonl │──▶│ 3 augmented views / train clip  │
 │ Svarah, LibriSpeech (English)│   │ speaker-disjoint    │   │ → data/features/*.npz (cache)   │
 └──────────────────────────────┘   │ train/val/test/ood  │   └───────────────┬─────────────────┘
                                    └─────────────────────┘                   ▼
                                                       ┌─────────────────────────────────────────┐
                                                       │ train HybridLID  (quantum, adjoint grad)│
                                                       │ train ClassicalLID (param-matched MLP)  │
                                                       │ early stopping on val macro-F1          │
                                                       └───────────────────┬─────────────────────┘
                                                                           ▼
                                 checkpoints/*.pt · results/REPORT.md · metrics · curves · *.qasm · benchmark
```

### 2.3 Concurrency model (live)

| Thread | Work | Blocking? |
|---|---|---|
| PortAudio callback | copy each 512-sample frame into `q_audio` | never (no allocation beyond a copy) |
| Main thread | assemble frames, run Silero VAD (CPU, < 1 ms/frame), cut utterances into `q_seg` | waits on `q_audio` |
| Worker thread | LID → ASR → MT on the GPU, print, log, TTS enqueue | waits on `q_seg` |
| TTS thread (optional) | pyttsx3 / SAPI speech, sets `busy` so the mic input is ignored while speaking | waits on its queue |

Capture never waits for GPU work, so no audio is dropped while a previous utterance is being translated.

---

## 3. Why the quantum model handles language detection

A quantum circuit simulated on a laptop can hold only about 10–20 qubits, i.e. a feature vector of about 10 values. Full speech recognition or translation needs millions of parameters and long sequences. A quantum circuit cannot do that today, on hardware or in simulation. The design puts the quantum circuit where it is **meaningful, measurable and on the critical path**:

* **Spoken language identification** is a compact classification problem: one utterance in, one of two labels out. A 10-dimensional projection of a good speech embedding carries enough information.
* It **controls the whole pipeline**: the LID decision chooses which ASR and which translation direction run. Running only one ASR model, instead of both and picking afterwards, roughly halves ASR compute. So the quantum component also contributes to the latency goal.
* It can be **compared fairly** against a classical head with the same number of parameters (§8.4). This is what makes the result scientific rather than decorative.
* It has a direct path to **real hardware**: the trained circuit is exported as OpenQASM 3 (§4.6).

All ASR and translation are done by pretrained AI4Bharat and Whisper models. This is stated openly as a design decision, not hidden.

---

## 4. Quantum model design

### 4.1 Hybrid LID head

```
embedding e ∈ R^1024 (Whisper-base, mean ⊕ std over valid frames)
  → LayerNorm(1024) → Dropout(0.1) → Linear(1024 → 10) → x = π · tanh(·)  ∈ (−π, π)^10
  → QuantumLayer: 10 qubits × 4 re-uploading layers → z_q = ⟨Z_q⟩ ∈ [−1, 1]^10
  → Linear(10 → 2) → softmax → P(hi), P(en)
```

### 4.2 Circuit ansatz: data re-uploading

For layers l = 0…L−1 and qubits q = 0…n−1:

```
          ┌──────────────────────┐ ┌────────┐ ┌────────┐ ┌────────┐
 q_0: ────┤ RY(s₀₀·x₀ + b₀₀)     ├─┤RZ(w⁰₀₀)├─┤RY(w¹₀₀)├─┤RZ(w²₀₀)├──■────────────X──  … repeat L times
          ├──────────────────────┤ ├────────┤ ├────────┤ ├────────┤  │            │
 q_1: ────┤ RY(s₀₁·x₁ + b₀₁)     ├─┤RZ(w⁰₀₁)├─┤RY(w¹₀₁)├─┤RZ(w²₀₁)├──X──■─────────┼──
          …                                                             …   CNOT ring
 q_9: ────┤ RY(s₀₉·x₉ + b₀₉)     ├─┤   …    ├─┤   …    ├─┤   …    ├─────────■─────■──
```

* **Encoding:** `RY(s_lq · x_q + b_lq)` with trainable scale s and bias b. The input is re-uploaded in every layer. Data re-uploading makes the model a trainable Fourier series in x whose frequency spectrum grows with L, so a single qubit can still represent non-linear decision boundaries.
* **Variational block:** `RZ·RY·RZ` = a general single-qubit rotation (Euler decomposition), 3 parameters per qubit per layer.
* **Entangler:** CNOT ring `(0→1, 1→2, …, 9→0)`, n CNOTs per layer, so every qubit's state influences the measured outputs within a few layers.
* **Readout:** exact expectation values ⟨Z_q⟩ for all 10 qubits, computed analytically from the statevector. No shot sampling, so no shot noise and no repeated runs.

| Quantity (n = 10, L = 4) | Value |
|---|---|
| Gates | 200 (160 rotations + 40 CNOT) |
| Angles per sample | 160 (layout `l·4n + block·n + q`) |
| Trainable circuit parameters | **200** = 5·n·L (40 scales + 40 biases + 120 rotation angles) |
| Hilbert-space dimension | 2¹⁰ = 1024 complex amplitudes (8 KB in complex64) |
| Whole hybrid head | **12,520** parameters (LayerNorm 2,048 + projection 10,250 + circuit 200 + output 22) |

### 4.3 Conventions (identical in every backend)

* qubit q ↔ bit q of the basis index (little-endian, same as Qiskit)
* `RX(θ)=e^{−iθX/2}`, `RY(θ)=e^{−iθY/2}`, `RZ(θ)=e^{−iθZ/2}`, `CNOT(control, target)`
* The circuit is a flat gate list (`type, q0, q1, angle_index`). This single description drives the torch, C++ and CUDA simulators and the QASM exporter (`qasr/quantum/circuit.py`).

### 4.4 Gradients: the adjoint method

The loss depends on the circuit through z_q = ⟨ψ|Z_q|ψ⟩. Back-propagation supplies g_q = ∂Loss/∂z_q. With the Hermitian observable O = Σ_q g_q Z_q and |ψ_k⟩ = U_k…U_1|0⟩:

```
λ ← O|ψ_G⟩                                   (O is diagonal: λ_i = ψ_i · Σ_q g_q (1 − 2·bit_q(i)))
for k = G … 1:
    if gate k is a rotation exp(−iθ_k P/2):
        ∂Loss/∂θ_k = 2·Re⟨λ| ∂U_k |ψ_{k−1}⟩ = Im ⟨λ| P |ψ_k⟩
    ψ ← U_k† ψ ,   λ ← U_k† λ
```

* **Cost:** one forward pass + about two statevector passes, **independent of the number of parameters**.
* **Parameter-shift alternative:** 2 circuit runs *per angle*, i.e. 320 full simulations per sample here, about 100× more work.
* **Memory:** O(B·2ⁿ). Torch autograd instead stores every intermediate state, O(G·B·2ⁿ): about 420 MB for G = 200, B = 256, n = 10.
* Gradients with respect to the encoding angles flow back into the classical projection, so the hybrid model is trained **end to end** with one optimiser.

### 4.5 Why 10 qubits and 4 layers

| n | 2ⁿ | statevector | shared mem (backward, 256 thr.) | comment |
|---|---|---|---|---|
| 8 | 256 | 2 KB | ≈ 4.5 KB | fastest; less capacity |
| **10** | 1024 | 8 KB | ≈ 17 KB | **default**: ≥ 5 resident blocks per SM on Ada |
| 12 | 4096 | 32 KB | ≈ 65 KB | needs > 48 KB opt-in; fewer resident blocks; the kernel's hard upper limit |

Ten qubits balance expressivity with simulation throughput and stay small enough to run on real hardware later. L = 4 gives a 4-frequency Fourier spectrum per input while keeping the circuit shallow (depth ≈ 4 × (1 + 3 + 10)).

### 4.6 Export to real hardware

`python -m qasr export-qasm` writes:

* `results/lid_circuit_parametric.qasm`: OpenQASM 3 with the 10 inputs as `input float[64] x0…x9` and the trained weights baked in as constants.
* `results/lid_circuit_sample.qasm`: fully bound for one real test clip. Load it in Qiskit with `qiskit.qasm3.loads`, transpile it for an IBM backend, and estimate ⟨Z_q⟩ from the measurement counts (P(0) − P(1)).

---

## 5. Low-level simulation engine (C++ / CUDA)

Three interchangeable backends implement the same interface, `expz = backend(angles[B, 160]) → [B, 10]`, with gradients:

| Backend | Language | Gradient | Where it runs | Purpose |
|---|---|---|---|---|
| `cuda_kernel` | CUDA C (`kernels/qsim_cuda.cu`), compiled at runtime by **NVRTC via CuPy** | adjoint, inside the kernel | NVIDIA GPU | **production / fastest** |
| `cpu_kernel` | C++17 + OpenMP (`kernels/qsim_cpu.cpp`), compiled by g++/clang/MSVC, loaded with `ctypes` | adjoint | any CPU | dev laptops, CPU inference |
| `torch` | PyTorch | autograd | CPU / GPU | reference + universal fallback |

### 5.1 CUDA kernel design

* **One thread block = one audio sample.** The grid size equals the batch size, so a training batch of 256 is simulated as 256 independent blocks.
* **The whole statevector lives in on-chip shared memory.** The forward pass keeps ψ (8 KB at n = 10); the backward pass keeps ψ and λ (16 KB). Inside the gate loop there is **zero global-memory traffic** except reading the angle.
* **Thread mapping:** a single-qubit gate acts on 2ⁿ⁻¹ independent amplitude pairs `(i₀, i₀ | 1<<q)`, where `i₀ = insert_zero(p, q)`. The threads stride over pairs. CNOT swaps pairs with control = 1. A `__syncthreads()` separates consecutive gates.
* **Reductions** (⟨Z_q⟩ and the gradient inner products) use a power-of-two shared-memory tree reduction. Only the imaginary part of ⟨λ|P|ψ⟩ is reduced, as derived in §4.4.
* **Backward recomputes the forward pass inside the kernel** instead of storing it. This trades about 1 extra pass for O(1) extra memory.
* **No toolchain needed:** NVRTC compiles the kernel the first time it runs (about 1 s). There is no `nvcc` and no Visual Studio, which matters on a Windows gaming laptop. Kernels launch on PyTorch's current CUDA stream (`cupy.cuda.ExternalStream`), and tensors are shared zero-copy through DLPack.
* **Occupancy estimate** (RTX 4070 Laptop, 36 SMs, up to 100 KB shared memory per SM): at n = 10 the backward pass uses about 17 KB per block, so about 5 blocks per SM can be resident, i.e. about 180 samples simulated concurrently.
* `--use_fast_math` is applied. Accuracy is still checked against the float32 reference by the self-test.

### 5.2 C++ CPU kernel

The same algorithm, written with `std::complex<float>`. It uses `#pragma omp parallel for` over the batch with a thread-local statevector, and is compiled with `-O3 -march=native -fopenmp` (AVX2 or AVX-512 auto-vectorisation). It is built automatically on first use and cached in `kernels/build/`, keyed by a hash of the source file.

### 5.3 Selection and safety net

With `quantum.backend: auto`, the order on GPU is `cuda_kernel → torch` and on CPU `cpu_kernel → torch`. Before a kernel backend is accepted it must pass a **numerical self-test**: random angles and random upstream gradients, compared with the torch reference. The tolerance is max |Δ| ≤ 2·10⁻⁴ for both the forward values and the gradients. A kernel that fails to compile or fails the test is skipped automatically and the next backend is used.

### 5.4 Measured results (development laptop, i7-1165G7, CPU only)

| Test | Result |
|---|---|
| torch simulator vs **Qiskit** `Statevector` (4 qubits, 2 layers, 3 random inputs) | max \|Δ⟨Z⟩\| ≤ **3.0·10⁻⁷** |
| C++ adjoint gradients vs torch autograd (10 qubits, 4 layers) | max \|Δ\| ≤ **3.4·10⁻⁷** |
| forward + backward, B = 256, n = 10, L = 4 | torch 3,164 ms · **C++ 239 ms (13× faster)** |
| forward, B = 1 (live-inference case) | torch 196 ms · **C++ 5.4 ms (36× faster)** |

The GPU numbers are produced on the RTX 4070 by `python -m qasr selftest` and written to `results/selftest.json`.

---

## 6. Data plan (AI4Bharat)

### 6.1 Sources

All of these repositories are gated ("auto" approval) on Hugging Face. Accepting their terms once per account is enough.

| Name in config | Repository / config / split | Lang | Full size | Used (max clips) | Role |
|---|---|---|---|---|---|
| `kathbath_hi` | ai4bharat/Kathbath · hindi · train | hi | 91,752 clips · 15.8 GB | 10,000 | read speech, many speakers |
| `indicvoices_hi` | ai4bharat/IndicVoices · hindi · train | hi | 445,160 clips · 40.5 GB | 12,000 | spontaneous / conversational, varied channels |
| `shrutilipi_hi` | ai4bharat/Shrutilipi · hindi · train | hi | 734,675 clips · 97.8 GB | 5,000 | broadcast news |
| `svarah_en` | ai4bharat/Svarah · test | en | 6,656 clips · 1.1 GB | all | **Indian-accented English** (in-domain for users) |
| `librispeech_en` | openslr/librispeech_asr · clean · train.100 | en | 28,539 clips · 6.6 GB | 20,000 | English volume / class balance |
| `lahaja_hi_ood` | ai4bharat/Lahaja · test | hi | 6,152 clips · 1.5 GB | 2,000 | **out-of-domain test only** (Hindi from many L1 backgrounds) |
| `librispeech_en_ood` | openslr/librispeech_asr · clean · test | en | 2,620 clips | 2,000 | **out-of-domain test only** (unseen speakers) |

The totals are about 27k Hindi and 27k English training/validation/test clips plus 4k out-of-domain clips: about 58k clips, about 90 h of audio, about 7 GB of FLAC. About 10–14 GB of parquet are streamed, instead of the 165 GB the full corpora would take.

AI4Bharat has no large *English* corpus, so LibriSpeech supplies volume. Svarah is up-weighted ×2 in the sampler so the model learns Indian-accented English. This matters because the users are Indian speakers.

### 6.2 Streaming acquisition (`qasr/data/download.py`)

* `datasets.load_dataset(..., streaming=True)` with `Audio(decode=False)`: only the rows we keep are fetched. Audio is decoded by **soundfile**, with a **PyAV** fallback that bundles FFmpeg, so there is no torchcodec or system FFmpeg dependency.
* `shuffle(seed, buffer=1000)` randomises the shard order and buffers rows, which gives diversity beyond the first shards.
* **Speaker cap:** at most 80 clips per speaker, which forces speaker diversity.
* **Quality filters:** duration ≥ 0.8 s, finite samples, not silent. Clips longer than 12 s are center-cropped before saving.
* **Normalisation:** mono, 16 kHz (polyphase `resample_poly`), 16-bit FLAC.
* **Resumable:** each clip ID is deterministic (`source_index` in the seeded stream), the manifest is appended one row at a time, and a restart skips existing IDs. Network errors are retried with back-off. Gated-access errors stop immediately with the exact URL to accept.

### 6.3 Splits

* **Speaker-disjoint** 90 / 5 / 5 train / val / test split: `md5(seed:speaker) mod 10000`, so no speaker appears in two splits. This prevents speaker-identity leakage from inflating accuracy.
* Svarah has no speaker ID, so the tuple (district, gender, age group, L1) serves as a speaker proxy. Shrutilipi has no speaker information, so its split is per clip.
* `test_ood` contains only Lahaja and LibriSpeech-test and is never used for training or model selection.

### 6.4 Manifest schema (`data/manifest.jsonl`)

```json
{"id": "kathbath_hi_0000042", "path": "clips/kathbath_hi/kathbath_hi_0000042.flac", "source": "kathbath_hi",
 "lang": "hi", "label": 0, "speaker": "kathbath_hi:1234", "split": "train", "duration": 5.12, "text": "…"}
```

---

## 7. Feature pipeline

| Item | Value |
|---|---|
| Encoder | `openai/whisper-base` encoder (frozen, about 20M parameters, fp16 on GPU) |
| Input | log-mel 80 × 3000 (30 s padded, standard Whisper front end) |
| Pooling | mean ⊕ std over the **valid** frames only (50 frames/s, padding masked), giving 1024-d |
| Training views | 3 per clip: random crop 1–6 s, gain ±6 dB, 50 % noise (white or brown, SNR 10–40 dB) |
| Eval views | 1 per clip: center crop of at most 6 s, no augmentation |
| Cache | `data/features/whisper_base_mean_std/{train,val,test,test_ood}.npz` (fp16; about 350 MB) |
| Ablation option | `features.encoder=mel_stats`: no neural encoder, mean/std of 80 log-mels (160-d) |

The encoder runs **once**. Training then reads the cached embeddings, so each epoch of quantum training touches only 1024-d vectors and takes seconds rather than hours. Multiple augmented views give robustness to crop length, gain and noise without paying for the encoder every epoch.

---

## 8. Training and evaluation protocol

### 8.1 Optimisation

| Hyper-parameter | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 (circuit parameters: none) |
| Learning rate | 3e-3 classical layers · **1e-2 circuit parameters** (separate parameter groups) |
| Schedule | 1-epoch linear warm-up → cosine decay |
| Batch | 256 · epochs ≤ 40 · early stopping on val macro-F1, patience 8 |
| Loss | cross-entropy, label smoothing 0.05 |
| Sampling | class-balanced with replacement; `svarah_en` × 2; one random view per clip per epoch |
| Regularisation | dropout 0.1, gradient clipping 1.0 |
| Precision | encoder fp16; simulator complex64 / float32 |
| Seed | 1234 (Python, NumPy, torch) |

### 8.2 Classical baseline (parameter-matched)

`ClassicalLID` is identical to the hybrid model except that the quantum layer is replaced by `Linear(10→h) → tanh → Linear(h→10) → tanh`, where h = round((5nL − n)/(2n + 1)) = 9. That gives **199 parameters vs 200** for the circuit, and 12,519 vs 12,520 parameters overall.

### 8.3 Metrics (`results/metrics_*.json`, `results/REPORT.md`)

* Accuracy, macro-F1, ROC-AUC and the confusion matrix on val, test and **test_ood**.
* Per-source accuracy, to expose dataset or channel bias. One source failing means the model learned recording conditions rather than language.
* Head latency at batch 1 on the training device, mean and p95.
* Training curves (`curves_*.png`) and epoch history.
* End-to-end benchmark (`results/benchmark.json`): LID accuracy, ASR WER/CER per language, per-stage latency, and 20 example translations.

### 8.4 Research questions this setup answers

1. Does a 200-parameter, 10-qubit circuit reach the same LID accuracy as a 199-parameter classical block on the same features?
2. Does it generalise *out of domain* (Lahaja, unseen LibriSpeech speakers) as well?
3. What does the quantum head cost in latency when simulated with a custom CUDA kernel, compared with PyTorch and C++?
4. Ablations (via `--set`): `quantum.n_qubits ∈ {6, 8, 10, 12}`, `quantum.n_layers ∈ {1, 2, 4, 6}`, `quantum.entangler ∈ {ring, line}`, `features.encoder=mel_stats` (a harder task, where the circuit has to do more of the work).

---

## 9. Live inference pipeline

### 9.1 Components

| Stage | Model / tool | Settings |
|---|---|---|
| Capture | `sounddevice` (PortAudio) | 16 kHz, mono, float32, block 512 (32 ms), `latency="low"` |
| VAD | Silero VAD v5 (`silero-vad`) | start ≥ 0.5 for 2 frames · end < 0.35 for 350 ms · pre-roll 300 ms · min speech 300 ms · max segment 8 s · keep 100 ms tail |
| LID | Whisper-base encoder + `HybridLID` | confidence threshold 0.65, otherwise keep the previous language (hysteresis for short or ambiguous utterances) |
| Hindi ASR | `ai4bharat/indic-conformer-600m-multilingual` | ONNX Runtime GPU 1.20.1, **CTC** decoding (`rnnt` optional) |
| Hindi ASR (alt.) | `ai4bharat/indicwav2vec-hindi` | wav2vec2 CTC, fp16 PyTorch, greedy |
| English ASR | `distil-whisper/distil-large-v3` | fp16, SDPA attention, greedy (`num_beams=1`), `language="en"` forced (skips Whisper's own language detection) |
| English ASR (alt.) | faster-whisper `distil-large-v3` | CTranslate2 float16 |
| MT hi→en | `ai4bharat/indictrans2-indic-en-dist-200M` | fp16, greedy, max 256 tokens, IndicTransToolkit pre/post-processing (built-in fallback) |
| MT en→hi | `ai4bharat/indictrans2-en-indic-dist-200M` | same |
| TTS | pyttsx3 → Windows SAPI | optional; microphone input ignored while speaking |
| Fallback LID | Whisper-base language token (`<|hi|>` vs `<|en|>` logits) | used automatically if no trained checkpoint exists |

### 9.2 Estimated VRAM budget (RTX 4070 Laptop, 8 GB)

| Model | Precision | ≈ VRAM |
|---|---|---|
| IndicConformer 600M (ONNX) | fp32 | 2.4 GB |
| distil-large-v3 (756M) | fp16 | 1.5 GB |
| IndicTrans2 200M × 2 | fp16 | 0.8 GB |
| Whisper-base (LID) | fp16 | 0.15 GB |
| Activations, CUDA context, ONNX arena | — | 1–1.5 GB |
| **Total** | | **≈ 6 GB** (fits in 8 GB) |

---

## 10. Latency engineering

### 10.1 Latency budget per utterance (targets; measured values go to `benchmark.json`)

| Stage | Budget | Notes |
|---|---|---|
| Frame granularity | 32 ms | Silero frame size |
| VAD compute | < 1 ms / frame | CPU, runs concurrently with the GPU worker |
| **End-of-utterance wait** | **350 ms** | `live.min_silence_ms`, the main trade-off knob (lower = faster but may cut sentences) |
| LID | 10–20 ms | Whisper-base encoder fp16 + quantum head (< 1 ms with the CUDA kernel) |
| ASR | 80–200 ms | depends on utterance length; CTC (Hindi) and greedy decoding (English) |
| MT | 40–120 ms | 200M distilled model, greedy |
| **Perceived total** | **≈ 0.5–0.75 s** after the speaker stops | goal N1 ≤ 1 s |

### 10.2 Optimisations implemented

1. **Route, don't duplicate:** LID chooses one ASR and one MT direction, so only one of each runs.
2. **Distilled models everywhere:** distil-large-v3 (2 decoder layers) and IndicTrans2-dist-200M (instead of 1.1B).
3. **Greedy decoding**, with the language forced in Whisper, CTC decoding for Hindi, and no timestamps.
4. **fp16 + SDPA attention + TF32 matmuls + `cudnn.benchmark`**.
5. **All models load once and are warmed up** at start-up with dummy audio and text, so the first utterance is not slow.
6. **Short, tunable endpointing** with pre-roll, so speech onsets are not clipped.
7. **Producer/consumer threads:** audio capture and VAD never wait for the GPU.
8. **Custom CUDA quantum kernel:** the quantum step adds less than 1 ms instead of tens of milliseconds with PyTorch on the GPU (many tiny kernel launches) or about 200 ms with PyTorch on the CPU.
9. **Hysteresis on LID confidence** avoids language flip-flopping on short words like "OK" or "haan".

### 10.3 Further latency options (not enabled by default)

* `live.min_silence_ms=250`: about 100 ms faster, with a higher risk of splitting sentences.
* `live.hindi_asr=indicwav2vec`: pure-PyTorch GPU path if ONNX Runtime falls back to the CPU.
* `live.english_asr_backend=faster_whisper`: CTranslate2 kernels.
* Streaming partial hypotheses during long utterances (future work, §16).

---

## 11. Software stack and versions

| Layer | Choice | Version |
|---|---|---|
| OS | Windows 11 (native); Linux / WSL2 supported via `*.sh` | — |
| Python | CPython | 3.11 (3.10–3.12 supported) |
| DL framework | PyTorch + torchaudio | 2.6.0 + CUDA 12.4 wheels |
| GPU kernels | CUDA C via NVRTC (`cupy-cuda12x`) | CuPy 13.x |
| CPU kernels | C++17, OpenMP | g++ / clang / MSVC (optional) |
| Models | transformers | 4.46.3 (pinned; IndicTrans2 remote code) |
| Hub | huggingface_hub (`hf` CLI) | ≥ 0.34, < 1.0 |
| Data | datasets (streaming parquet) | ≥ 2.21, < 4.0 |
| ONNX | onnxruntime-gpu | 1.20.1 (as the IndicConformer model card requires) |
| Audio | soundfile, PyAV, sounddevice, silero-vad | latest |
| Metrics | scikit-learn, jiwer | latest |
| Quantum reference / export | Qiskit (optional, for running the exported QASM) | any 1.x / 2.x |

---

## 12. Project structure

```
Quantum ASR/
├── INSTRUCTIONS.md              step-by-step guide for the training machine
├── README.md
├── docs/TECHNICAL_DESIGN.md     this document
├── configs/default.yaml         every parameter (override with --set key=value)
├── requirements.txt
├── setup.ps1 | setup.sh         one-time environment setup + self-test + HF login
├── train.ps1 | train.sh         ONE command: download → features → train → evaluate → export → benchmark
├── live.ps1  | live.sh          live microphone translation
└── qasr/
    ├── __main__.py              CLI (all, selftest, download, features, train, live, benchmark, …)
    ├── config.py · utils.py · audio_io.py · access.py
    ├── data/download.py         HF streaming → FLAC + manifest, speaker-disjoint splits
    ├── features.py              Whisper / mel-stats embedders, augmented views, cache
    ├── models.py                HybridLID, ClassicalLID (parameter-matched)
    ├── train.py                 training loop, metrics, report, latency
    ├── benchmark.py             end-to-end LID / WER / latency benchmark
    ├── quantum/
    │   ├── circuit.py           gate-list circuit description (single source of truth)
    │   ├── torch_sim.py         reference batched statevector simulator
    │   ├── backends.py          cuda_kernel / cpu_kernel / torch + self-test + auto-selection
    │   ├── layer.py             QuantumLayer (nn.Module)
    │   ├── export_qasm.py       OpenQASM 3 export
    │   └── kernels/qsim_cuda.cu · qsim_cpu.cpp
    └── live/
        ├── vad.py · lid.py · asr.py · translate.py · tts.py
        └── pipeline.py          threads, routing, printing, session logs
```

Generated at run time (git-ignored): `.venv/`, `data/`, `checkpoints/`, `results/`, `qasr/quantum/kernels/build/`.

---

## 13. Configuration reference

The most important keys in `configs/default.yaml`:

| Key | Default | Effect |
|---|---|---|
| `data.sources[*].max_items` | see §6.1 | clips per source (scale all with `--scale 0.5`) |
| `data.max_per_speaker` | 80 | speaker-diversity cap |
| `features.encoder` | whisper_base | or `mel_stats` (ablation) |
| `features.views_train` | 3 | augmented crops per training clip |
| `quantum.n_qubits` / `n_layers` | 10 / 4 | circuit size (n ≤ 12) |
| `quantum.entangler` | ring | or `line` |
| `quantum.backend` | auto | `cuda_kernel`, `cpu_kernel`, `torch` |
| `train.epochs` / `patience` | 40 / 8 | training length / early stopping |
| `train.lr` / `lr_quantum` | 3e-3 / 1e-2 | learning rates |
| `train.source_weight.svarah_en` | 2.0 | Indian-English up-weighting |
| `live.min_silence_ms` | 350 | endpointing latency |
| `live.lid_min_confidence` | 0.65 | hysteresis threshold |
| `live.hindi_asr` | indicconformer | or `indicwav2vec` |
| `live.hindi_asr_decoding` | ctc | or `rnnt` |
| `live.english_asr_backend` | transformers | or `faster_whisper` |
| `live.mt_num_beams` | 1 | >1 = better translations, slower |

---

## 14. Verification and testing

| Check | How | Status |
|---|---|---|
| Simulator correctness vs Qiskit | random circuits, ⟨Z_q⟩ compared | ✅ ≤ 3·10⁻⁷ (dev laptop) |
| Adjoint gradients (C++) vs autograd | random angles and upstream gradients | ✅ ≤ 3.4·10⁻⁷ |
| CUDA kernel correctness | same self-test, run automatically on the GPU machine before use; automatic fallback if it fails | ⏳ runs on the RTX 4070 (`selftest.json`) |
| End-to-end training path | `python -m qasr smoketest`: synthetic audio → features → hybrid + classical training → report → QASM → VAD | ✅ passes on CPU (dev laptop) |
| HF access to all gated repos | `python -m qasr check-access` | ⏳ needs the training account |
| Full data + training | `train.ps1 --quick`, then `train.ps1` | ⏳ RTX 4070 |
| Live pipeline + latency | `benchmark` (automatic) and `live.ps1` | ⏳ RTX 4070 |

The development laptop has no NVIDIA GPU and no access to the gated models. So the CUDA kernel, the AI4Bharat models and the live path are verified on the training machine by the built-in self-test and benchmark. They were not run during development.

---

## 15. Risks, limitations and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LID learns *dataset/channel* differences instead of language (e.g. LibriSpeech audiobook vs Kathbath phone audio) | high accuracy in testing, failures live | multiple sources per class, Indian-English Svarah up-weighted, gain/noise/crop augmentation, per-source accuracy and out-of-domain tests in the report |
| Code-mixed Hinglish | LID has to pick one language | confidence hysteresis; IndicConformer transcribes English words in Devanagari; true code-switch handling is future work |
| No quantum *advantage* claim | — | the project claims a working hybrid system and a fair comparison, not a speed-up over classical computing |
| Gated repos / remote code changes upstream | download or model-load failures | pinned versions, fail-fast access check, alternative backends (`indicwav2vec`, `faster_whisper`), MT retry without KV cache |
| onnxruntime-gpu cannot find CUDA DLLs on Windows | Hindi ASR falls back to CPU (slower) | torch's bundled CUDA/cuDNN DLL directory is added at runtime; `indicwav2vec` alternative |
| Laptop thermal throttling during long runs | slower training | performance mode, mains power; every stage is resumable |
| CuPy/NVRTC unavailable | no custom kernel | automatic fallback to the torch backend (correct, slower) |
| 8 GB VRAM | out-of-memory with all models | budget about 6 GB (§9.2); batch sizes configurable |

---

## 16. Implementation plan and future work

### 16.1 Phases

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Architecture and technical design (this document) | ✅ |
| 1 | Codebase: data, features, quantum engine (torch/C++/CUDA), training, live pipeline, scripts, docs | ✅ built, CPU smoke-tested |
| 2 | Environment setup and self-test on the RTX 4070 (`setup.ps1`) | ⏳ |
| 3 | Trial run (`train.ps1 --quick`) to validate data access and the GPU path | ⏳ |
| 4 | Full training + report (`train.ps1`) | ⏳ |
| 5 | Live evaluation, latency tuning (`live.ps1`, `benchmark.json`) | ⏳ |
| 6 | Ablations (§8.4) and write-up | ⏳ |

### 16.2 Future work

1. **Streaming partial results:** show interim transcripts every about 1 s during long utterances.
2. **Quantum layer inside ASR:** insert a batched quantum layer before the CTC output of a fine-tuned Hindi model. The CUDA kernel already supports large batches (about 50 frames/s).
3. **Code-switch-aware LID:** 3 classes (hi / en / mixed) or frame-level LID.
4. **Noise-aware training:** add depolarising/readout noise channels to the simulator, so results transfer to real hardware.
5. **Run the exported circuit on IBM Quantum** and compare hardware against the simulator.
6. **Newer models:** the IndicConformer and IndicTrans2 model cards now point to *Indic-Transcribe* and *Indic-Translate* (bodhan-ai) as higher-quality successors. These can replace the current models via `configs/default.yaml` once evaluated.
7. **More languages:** IndicConformer and IndicTrans2 cover 22 Indian languages. Extending LID to n classes only changes `N_CLASSES` and the data sources.
