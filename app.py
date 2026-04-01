import os
import re
import uuid
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from transformers import pipeline
from gtts import gTTS
from pydub import AudioSegment
from openai import OpenAI
import nltk

# -------------------------------------------------------------------
# PROFESSIONAL SETUP & LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

app = FastAPI(title="AI Sales Engine - Ultimate Edition")

logger.info("Loading DistilRoBERTa Emotion Model...")
try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception as e:
    logger.error(f"Failed to load emotion model: {e}")
    raise

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


# -------------------------------------------------------------------
# CHALLENGE 1: THE EMPATHY ENGINE 
# -------------------------------------------------------------------
def process_empathy_engine(text: str) -> Dict[str, Any]:
    classification = emotion_classifier(text)[0]
    emotion = classification['label']
    score = classification['score']
    
    session_id = str(uuid.uuid4())
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{session_id}.mp3")
    final_file = os.path.join(OUTPUT_DIR, f"final_{session_id}.mp3")
    
    try:
        gTTS(text=text, lang='en', slow=False).save(temp_file)
        sound = AudioSegment.from_mp3(temp_file)
        base_rate = sound.frame_rate
        
        if emotion in ['joy', 'surprise']:
            new_rate = int(base_rate * 1.15)
            sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(base_rate)
            sound = sound + 3  
        elif emotion in ['sadness']:
            new_rate = int(base_rate * 0.85)
            sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(base_rate)
            sound = sound - 4  
        elif emotion in ['anger']:
            new_rate = int(base_rate * 0.95)
            sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(base_rate)
            sound = sound + 6  
        elif emotion in ['fear']:
            new_rate = int(base_rate * 1.2)
            sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(base_rate)
            sound = sound - 2
            
        sound.export(final_file, format="mp3")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    return {
        "emotion": emotion.capitalize(),
        "confidence": round(score * 100, 2),
        "audio_path": f"/audio/final_{session_id}.mp3"
    }


# -------------------------------------------------------------------
# CHALLENGE 2: THE PITCH VISUALIZER (Optimized Prompting)
# -------------------------------------------------------------------
def process_pitch_visualizer(text: str, style: str) -> List[Dict[str, str]]:
    scenes = nltk.tokenize.sent_tokenize(text)
    if len(scenes) < 3:
        scenes = [s.strip() for s in re.split(r'[;:]', text) if len(s.strip()) > 5]
    if not scenes:
        scenes = [text]

    storyboard = []
    
    for scene in scenes:
        enhanced_prompt = f"[{style}] Synthesized Visual Representation: {scene}"
        seed = abs(hash(scene)) % 10000 
        image_url = f"https://picsum.photos/seed/{seed}/1024/1024"
        
        if client:
            try:
                # NEW STRICT PROMPT INSTRUCTION TO FORCE DALL-E 3 COMPLIANCE
                system_instruction = (
                    f"You are a strict DALL-E 3 prompt engineer. "
                    f"The user wants an image in EXACTLY this style: '{style}'. "
                    f"Create a prompt that STARTS with this style description and describes the scene: '{scene}'. "
                    f"If the style is vector or 2D, explicitly append 'NO 3D, NO photorealism, flat colors only' to the prompt. "
                    f"Keep it under 400 characters."
                )
                
                prompt_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": scene}
                    ]
                )
                enhanced_prompt = prompt_response.choices[0].message.content
                
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    n=1,
                )
                image_url = image_response.data[0].url
            except Exception as e:
                logger.warning(f"API Fallback Triggered. Error: {e}")
                enhanced_prompt = f"(Fallback Mode) {scene}"

        storyboard.append({
            "scene_text": scene,
            "engineered_prompt": enhanced_prompt,
            "image_url": image_url
        })

    return storyboard


# -------------------------------------------------------------------
# WEB UI & ROUTING (With Animations & Loaders)
# -------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sales Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; min-height: 100vh; }}
        
        /* Animated Gradient Background */
        .bg-animate {{
            background: linear-gradient(-45deg, #f8fafc, #e0e7ff, #ede9fe, #f1f5f9);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }}
        @keyframes gradientBG {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        /* Glassmorphism Cards */
        .glass-card {{ 
            background: rgba(255, 255, 255, 0.85); 
            backdrop-filter: blur(12px); 
            border: 1px solid rgba(255, 255, 255, 0.6); 
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        }}

        /* Fade In Up Animation for Results */
        .fade-in-up {{
            opacity: 0;
            transform: translateY(30px);
            animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        @keyframes fadeInUp {{
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Spinner Animation */
        .spinner {{
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid #4f46e5;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        
        audio::-webkit-media-controls-panel {{ background-color: #f8fafc; }}
    </style>
</head>
<body class="bg-animate text-slate-800 p-6 md:p-12 relative">

    <div id="loading-overlay" class="hidden fixed inset-0 z-50 bg-white/80 backdrop-blur-md flex-col items-center justify-center">
        <div class="spinner mb-6"></div>
        <h2 id="loading-text" class="text-2xl font-bold text-indigo-600 tracking-tight">Initializing Engines...</h2>
        <p class="text-slate-500 mt-2 text-sm font-medium">This usually takes 5-10 seconds</p>
    </div>

    <div class="max-w-6xl mx-auto relative z-10">
        <header class="text-center mb-16 mt-4 fade-in-up" style="animation-delay: 0.1s;">
            <div class="inline-block mb-3 px-4 py-1.5 rounded-full bg-indigo-100/50 border border-indigo-200 text-indigo-700 font-semibold text-xs tracking-widest uppercase">
                Prototype v2.0
            </div>
            <h1 class="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 tracking-tight mb-4">
                Empathy & Vision Engine
            </h1>
            <p class="text-lg text-slate-600 font-light max-w-2xl mx-auto">
                Transforming unstructured text into emotionally intelligent audio and cinematic visual storyboards.
            </p>
        </header>
        
        <form action="/generate" method="post" onsubmit="showLoader()" class="glass-card p-8 md:p-10 rounded-3xl mb-16 fade-in-up transition-transform duration-300 hover:scale-[1.01]" style="animation-delay: 0.2s;">
            <div class="mb-8">
                <label class="block text-sm font-bold text-slate-700 mb-3 tracking-wider uppercase">Narrative Context / Script</label>
                <textarea name="user_text" rows="4" required class="w-full p-5 bg-white/70 border border-slate-200 rounded-2xl focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all outline-none resize-none shadow-inner text-lg placeholder-slate-400" placeholder="Type a highly emotional or dramatic sales story here..."></textarea>
            </div>
            
            <div class="mb-10">
                <label class="block text-sm font-bold text-slate-700 mb-3 tracking-wider uppercase">Generative Visual Style</label>
                <select name="style" class="w-full p-4 bg-white/70 border border-slate-200 rounded-2xl focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all outline-none appearance-none cursor-pointer text-slate-700 font-medium">
                    <option value="Cinematic photography, photorealistic, 8k resolution, dramatic volumetric lighting, highly detailed">Cinematic & Photorealistic</option>
                    <option value="Flat 2D vector illustration graphic design, solid pastel colors, minimal clean UI style, NO shading, NO 3D elements, pure white background">Corporate Minimalist Vector (2D Flat)</option>
                    <option value="3D isometric render, orthographic projection, soft studio lighting, clean minimal clay render style, pastel colors, tilt-shift lens">3D Isometric Studio</option>
                </select>
            </div>
            
            <button type="submit" class="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold text-lg py-4 px-6 rounded-2xl hover:from-indigo-700 hover:to-purple-700 shadow-[0_10px_20px_rgba(79,70,229,0.3)] hover:shadow-[0_15px_25px_rgba(79,70,229,0.4)] hover:-translate-y-1 transition-all duration-300 flex justify-center items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                Synthesize Media Pipeline
            </button>
        </form>

        {results_html}
    </div>

    <script>
        function showLoader() {{
            const overlay = document.getElementById('loading-overlay');
            overlay.classList.remove('hidden');
            overlay.classList.add('flex');
            
            const texts = [
                "Analyzing emotional tone...", 
                "Modulating vocal parameters...", 
                "Segmenting narrative arcs...", 
                "Engineering LLM visual prompts...",
                "Rendering final media..."
            ];
            let i = 0;
            const textElement = document.getElementById('loading-text');
            setInterval(() => {{
                i = (i + 1) % texts.length;
                textElement.innerText = texts[i];
            }}, 2000);
        }}
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTML_TEMPLATE.format(results_html="")

@app.post("/generate", response_class=HTMLResponse)
def handle_generation(user_text: str = Form(...), style: str = Form(...)):
    audio_data = process_empathy_engine(user_text)
    storyboard_data = process_pitch_visualizer(user_text, style)
    
    audio_html = f"""
    <div class="glass-card p-8 rounded-3xl mb-16 border-l-8 border-indigo-500 relative overflow-hidden fade-in-up" style="animation-delay: 0.1s;">
        <div class="absolute top-0 right-0 -mt-10 -mr-10 w-40 h-40 bg-indigo-400 opacity-20 rounded-full blur-3xl"></div>
        <div class="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4 relative z-10">
            <h2 class="text-3xl font-bold text-slate-800 tracking-tight flex items-center gap-3">
                <span>🎙️</span> Empathy Synthesis
            </h2>
            <div class="flex items-center gap-3 bg-white/60 p-2 rounded-2xl border border-slate-100">
                <span class="text-xs font-bold text-slate-500 uppercase tracking-widest pl-2">Emotion Detedted:</span>
                <span class="bg-gradient-to-r from-indigo-500 to-purple-500 text-white py-1.5 px-4 rounded-xl text-sm font-bold shadow-md">
                    {audio_data['emotion']} ({audio_data['confidence']}%)
                </span>
            </div>
        </div>
        <div class="bg-white/50 p-4 rounded-2xl border border-slate-100 relative z-10">
            <audio controls class="w-full rounded-xl"><source src="{audio_data['audio_path']}" type="audio/mpeg"></audio>
        </div>
    </div>
    """
    
    panels_html = ""
    for idx, panel in enumerate(storyboard_data):
        img_tag = f'<img src="{panel["image_url"]}" alt="Panel" class="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-1000">'
        panels_html += f"""
        <div class="glass-card rounded-3xl overflow-hidden flex flex-col group hover:-translate-y-3 hover:shadow-2xl transition-all duration-500 border border-slate-100 fade-in-up" style="animation-delay: {0.2 + (idx * 0.1)}s;">
            <div class="relative overflow-hidden h-64 bg-slate-200">
                {img_tag}
                <div class="absolute top-4 left-4 bg-white/95 backdrop-blur text-indigo-700 text-xs font-extrabold px-4 py-2 rounded-full shadow-lg tracking-wider uppercase">Scene {idx + 1}</div>
            </div>
            <div class="p-6 md:p-8 flex-grow flex flex-col bg-white/40">
                <p class="text-slate-800 font-semibold text-lg mb-6 leading-relaxed flex-grow border-l-4 border-indigo-200 pl-4">"{panel['scene_text']}"</p>
                <div class="mt-auto">
                    <p class="text-[10px] text-slate-500 font-mono bg-slate-100/80 p-4 rounded-xl border border-slate-200 overflow-hidden text-ellipsis line-clamp-3 leading-relaxed" title="{panel['engineered_prompt']}">
                        <strong class="text-indigo-500 uppercase tracking-widest block mb-1">Generated Prompt:</strong>
                        {panel['engineered_prompt']}
                    </p>
                </div>
            </div>
        </div>
        """
        
    storyboard_html = f'<div class="mb-12"><h2 class="text-3xl font-bold text-slate-800 mb-8 pl-2 tracking-tight fade-in-up flex items-center gap-3"><span>🖼️</span> Dynamic Storyboard</h2><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">{panels_html}</div></div>'
    
    return HTML_TEMPLATE.format(results_html=audio_html + storyboard_html)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(os.path.join(OUTPUT_DIR, filename))

if __name__ == "__main__":
    import uvicorn
    logger.info("Server running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)