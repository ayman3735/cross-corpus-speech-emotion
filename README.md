# Speech emotion recognition as a proxy for stress detection: MFCC vs. frozen WavLM

Code accompanying the paper *Cross-domain generalisation of pretrained speech representations for stress detection* (acted-emotion proxy study on RAVDESS, CREMA-D and EMO-DB).

All experiments are in one Colab notebook: `speech_emotion_proxy_experiments.ipynb`.

## What the notebook does
- Downloads RAVDESS (Zenodo), EMO-DB (Zenodo) and the full CREMA-D corpus (GitLab mirror via git-lfs) inside Colab.
- Extracts 82-dimensional MFCC/delta/ZCR/RMS features, 768-dimensional mean-pooled WavLM embeddings (frozen `microsoft/wavlm-base`, 16 kHz mono) and eGeMAPS (openSMILE, RAVDESS only).
- Runs speaker-independent (actor-grouped) evaluation with SVM and logistic regression: RQ1 feature comparison and controls, RQ2 training-actor reduction, RQ3/RQ4 cross-corpus and leave-one-corpus-out transfer with and without per-corpus z-score standardisation.
- Seed: 42.

## Running
1. Open the notebook in Google Colab and select a GPU runtime.
2. Run cells from top to bottom. Feature extraction is slow, so features are cached to `FEATURE_DIR` and reloaded on later runs.

## Notes
- Datasets are not redistributed; obtain them under their own licences.
- Per-corpus z-score standardisation uses unlabelled target-corpus statistics and is therefore transductive.
- All results concern acted emotion, not clinical stress or depression.
