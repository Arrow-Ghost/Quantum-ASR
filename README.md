# Quantum ASR

Live **Hindi ↔ English speech translation** with automatic language detection. The language detector is a **hybrid quantum–classical model**: a 10-qubit variational circuit simulated by a custom CUDA/C++ statevector engine with adjoint gradients.

```
mic → Silero VAD → [Whisper-base embedding → 10-qubit circuit] → hi | en
        hi → IndicConformer (AI4Bharat) → IndicTrans2 hi→en
        en → distil-whisper large-v3    → IndicTrans2 en→hi
```

- **Running it on the GPU laptop:** see [INSTRUCTIONS.md](INSTRUCTIONS.md)
- **Architecture, parameters and design rationale:** see [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)

Quick reference:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # once
.\train.ps1 --quick                                    # 30-min trial
.\train.ps1                                            # full: data → train → evaluate → benchmark
.\live.ps1                                             # live translation
```

To check the whole training path on any CPU machine without downloading anything:

```powershell
python -m qasr smoketest
```
