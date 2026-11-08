# Quantum ASR – instructions for the training machine

**What this project does:** listens to live speech, works out whether it is **Hindi or English**, transcribes it, and **translates it into the other language** (Hindi → English, English → Hindi) within about a second.
The language detection runs on a **hybrid quantum–classical model**: a 10-qubit variational circuit simulated on your GPU by a custom CUDA kernel. You train that model here.

Every step is one command. If one breaks, the [Troubleshooting](#7-troubleshooting) table at the end covers the common problems.

| | |
|---|---|
| Target machine | HP Victus, Intel i5-H, NVIDIA RTX 4070 Laptop GPU, Windows 11 |
| Your time needed | about 20 minutes of setup; everything after that runs unattended |
| Machine time | trial run about 30 min · full run about 2–4 h (mostly downloading) |
| Disk space | **at least 60 GB free** (data about 8 GB, models about 8 GB, Python packages about 8 GB, plus headroom) |
| Internet downloaded | about 10–14 GB of audio + about 7 GB of models |

---

## 1. One-time preparation (≈10 min)

1. **Copy this whole folder** to a fast local drive, for example `C:\QuantumASR` or `D:\QuantumASR`.
   Avoid OneDrive or Google Drive synced folders, because several GB of data will be written into it.
2. **Update the NVIDIA driver** (GeForce Experience → Drivers, or nvidia.com). Check it in a terminal:
   ```powershell
   nvidia-smi
   ```
   You should see "NVIDIA GeForce RTX 4070 Laptop GPU".
3. **Install Python 3.11** from <https://www.python.org/downloads/release/python-3119/> ("Windows installer 64-bit").
   On the first installer screen, tick **"Add python.exe to PATH"**.
4. **Create a free Hugging Face account** at <https://huggingface.co/join>.
5. While logged in, open **each link below** and click **"Agree and access repository"**. Access is granted instantly.
   AI4Bharat requires this for every dataset and model.

   Datasets:
   - <https://huggingface.co/datasets/ai4bharat/Kathbath>
   - <https://huggingface.co/datasets/ai4bharat/IndicVoices>
   - <https://huggingface.co/datasets/ai4bharat/Shrutilipi>
   - <https://huggingface.co/datasets/ai4bharat/Svarah>
   - <https://huggingface.co/datasets/ai4bharat/Lahaja>

   Models:
   - <https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual>
   - <https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M>
   - <https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M>
   - <https://huggingface.co/ai4bharat/indicwav2vec-hindi> (backup Hindi ASR)

6. Create an access token: <https://huggingface.co/settings/tokens> → **Create new token** → type **Read** → copy it. You'll paste it in step 2.
7. Laptop settings for long runs:
   - Keep it **plugged in**.
   - Set Windows power mode to **Best performance**, and set the Victus/OMEN Gaming Hub to **Performance**.
   - Set **Sleep** to **Never** while plugged in (Settings → System → Power).

## 2. Setup (≈10 min, one command)

Open **PowerShell** in the project folder (right-click the folder → "Open in Terminal") and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This will:

1. Create a private Python environment in `.venv`.
2. Install PyTorch with CUDA and every other requirement.
3. Install CuPy, which runs the custom CUDA quantum kernel.
4. Run a **self-test**: GPU detection, plus a check that the C++ and CUDA quantum simulators match the reference to within 1e-4.
5. Ask for your Hugging Face token (paste it; nothing is shown while you type).
6. Check that you have access to every dataset and model above.

Check the output for:

- `quantum backend cuda_kernel passed self-test`. This is the fast path. If it says *unavailable*, training still works on the slower torch backend.
- `[ok]` next to every repository. A `[GATED]` line means you missed the "Agree" button for that link in step 1.5.

## 3. Trial run (≈30 min), to confirm everything works

```powershell
.\train.ps1 --quick
```

This uses 5 % of the data and 5 epochs, so it runs every stage quickly: download, features, training, evaluation, OpenQASM export and benchmark. It should finish with `ALL DONE`.

## 4. Full training run (≈2–4 h, one command)

```powershell
.\train.ps1
```

The stages, in order:

| # | Stage | What happens | Approx. time on RTX 4070 |
|---|---|---|---|
| 1 | access check + self-test | as in setup | 1 min |
| 2 | download | streams about 58k clips (27k Hindi, 27k English, 4k out-of-domain test) from AI4Bharat + LibriSpeech, saved as 16 kHz FLAC in `data/clips` | 1–3 h (depends on internet) |
| 3 | features | Whisper-base encoder turns every clip into a 1024-number embedding (3 augmented crops per training clip) | 20–40 min |
| 4 | train hybrid | 10-qubit quantum circuit + classical layers, about 12.5k parameters | 10–30 min |
| 5 | train classical | same model with an MLP in place of the circuit (fair baseline) | 2–5 min |
| 6 | report + QASM | `results/REPORT.md`, OpenQASM 3 files of the trained circuit | seconds |
| 7 | benchmark | downloads the live models, then runs 100 held-out clips end to end (LID accuracy, ASR error rate, latency) | 5–10 min |

**If it stops** (internet drop, laptop restart), run `.\train.ps1` again. Every stage **resumes** where it left off: downloaded clips and cached features are reused.

## 5. Live translation demo

```powershell
.\live.ps1
```

Speak Hindi or English into the microphone. After each sentence (when you pause for about 0.35 s) you'll see something like:

```
[HI->EN | LID 0.98 | audio 2.4s | compute 310 ms | ~660 ms after you stopped]
  मैं कल दिल्ली जा रहा हूँ
  I am going to Delhi tomorrow
```

Options:

```powershell
.\live.ps1 devices              # list microphones
.\live.ps1 --device 2           # use microphone number 2
.\live.ps1 --speak              # also speak the translation aloud
.\live.ps1 --file sample.wav    # run a recording through the same pipeline
```

Press **Ctrl + C** to stop. A per-stage latency table is printed, and the session is saved in `results/live_sessions/`.

Tips:

- Use **headphones** with `--speak`, otherwise the laptop hears its own voice. The program also mutes the mic while speaking.
- Hindi speech output needs a Hindi Windows voice: Settings → Time & language → Speech → **Add voices** → Hindi.
- Use **Windows Terminal**, the default on Windows 11, so Devanagari text displays correctly.

## 6. What to send back

Zip and send these two folders. Do **not** send `data/` or `.venv/`, which are huge.

- `results/` contains:
  - `REPORT.md` (quantum vs classical comparison)
  - `metrics_*.json`, `curves_*.png`
  - `selftest.json` (GPU kernel speed)
  - `benchmark.json` (latency, accuracy)
  - `lid_circuit_*.qasm`
  - `logs/`
- `checkpoints/` contains `hybrid_best.pt` and `classical_best.pt` (small files)

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `running scripts is disabled on this system` | Use the exact command from step 2: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`. For the other scripts, run `powershell -ExecutionPolicy Bypass -File .\train.ps1` |
| `[GATED] ai4bharat/...` or `401 / 403` errors | Open that repository link while logged in and click **Agree and access repository**. Then run `.\.venv\Scripts\hf.exe auth login` again |
| `cuda_available False` in the self-test | Update the NVIDIA driver and reboot. Make sure setup installed the `+cu124` torch build (`.\.venv\Scripts\python -c "import torch; print(torch.__version__)"`) |
| `cuda_kernel unavailable` | Training falls back to the torch backend automatically. To retry: `.\.venv\Scripts\pip install cupy-cuda12x`, then `.\.venv\Scripts\python -m qasr selftest` |
| `CUDA out of memory` | Close games and browsers. Then use smaller batches: `.\train.ps1 --set features.batch_size=32 --set train.batch_size=128` |
| Download very slow or stuck | Leave it running; it retries by itself. To download less: `.\train.ps1 --scale 0.5` |
| Hindi ASR slow (more than 1 s) in live mode | onnxruntime fell back to the CPU. Switch the Hindi model: `.\live.ps1 --set live.hindi_asr=indicwav2vec` |
| Translation error mentioning `past_key_values` / cache | The code retries without the cache automatically. If it still fails, report the error |
| Wrong microphone | `.\live.ps1 devices`, then `.\live.ps1 --device N` |
| Want lower latency | `--set live.min_silence_ms=250` (cuts sentences sooner) or `--set live.hindi_asr_decoding=ctc` (already the default) |

Everything is configured in `configs/default.yaml`. Any value can be overridden with `--set key=value`.
The technical design is in `docs/TECHNICAL_DESIGN.md`.
