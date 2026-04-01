# 🚀 Empathy & Vision Engine (AI Sales Assistant)

A high-performance, modular AI pipeline designed to transform unstructured sales narratives into emotionally intelligent synthesized audio and cinematic visual storyboards. 

Built as a Proof of Concept (POC) for the AI Sales Assistant integration.

## 🌟 Core Architecture & Design Choices

1. **Graceful API Degradation (Circuit Breaker):** Challenge 2 integrates `gpt-3.5-turbo` (Prompt Engineering) and `dall-e-3` (Image Generation). To ensure the application remains strictly evaluable in the event of an `HTTP 429 Insufficient Quota` error, a graceful fallback mechanism intercepts the exception and serves deterministic, programmatically seeded placeholder imagery. The UI and narrative segmentation (`nltk`) pipeline remain uninterrupted.
2. **Dynamic Audio Modulation:** Challenge 1 utilizes `DistilRoBERTa` for granular emotion classification (7-way). Instead of relying on expensive TTS APIs, base audio is generated locally via `gTTS` and programmatically modulated (Pitch, Speed, Volume) in real-time using `pydub` based on a custom semantic-to-acoustic mapping matrix.
3. **Thread-Safe Concurrency:** The FastAPI backend uses synchronous `def` endpoints for heavy I/O operations (audio modulation, synchronous OpenAI calls), allowing FastAPI to automatically delegate tasks to a background threadpool, preventing main-loop blocking. Temporary file management uses `uuid` mapping to prevent race conditions.

## 🛠️ Technical Stack
* **Backend:** Python 3.9+, FastAPI, Uvicorn
* **ML/NLP:** Hugging Face `transformers`, NLTK
* **Audio Processing:** `gTTS`, `pydub`, `ffmpeg`
* **Generative AI:** OpenAI API
* **Frontend:** HTML5, Tailwind CSS, Glassmorphism UI

## ⚙️ Setup & Execution

1. **Install System Dependencies:** Ensure `ffmpeg` is installed on your machine (`brew install ffmpeg` for macOS).
2. **Install Python Libraries:** ```bash
   pip install -r requirements.txt