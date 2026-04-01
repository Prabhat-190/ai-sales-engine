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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

app = FastAPI(title="AI Sales Engine")

try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

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

def process_pitch_visualizer(text: str, style: str) -> List[Dict[str, str]]:
    scenes = nltk.tokenize.sent_tokenize(text)
    if len(scenes) < 3:
        scenes = [s.strip() for s in re.split(r'[;:]', text) if len(s.strip()) > 5]
    if not scenes:
        scenes = [text]

    storyboard = []
    
    for scene in scenes:
        enhanced_prompt = f"[{style}] Scene: {scene}"
        seed = abs(hash(scene)) % 10000 
        image_url = f"https://picsum.photos/seed/{seed}/1024/1024"
        
        if client:
            try:
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
                logger.warning(f"API Fallback Triggered: {e}")
                enhanced_prompt = f"(Fallback Mode) {scene}"

        storyboard.append({
            "scene_text": scene,
            "engineered_prompt": enhanced_prompt,
            "image_url": image_url
        })

    return storyboard

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sales Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            margin: 0; 
            min-height: 100vh;
            background-color: #f8fafc;
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,0.05) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,0.05) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,0.05) 0, transparent 50%);
            background-attachment: fixed;
        }}
        .glass-panel {{ 
            background: rgba(255, 255, 255, 0.7); 
            backdrop-filter: blur(16px); 
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.5); 
            box-shadow: 0 10px 40px -10px rgba(0,0,0,0.08);
        }}
        .animate-up {{
            opacity: 0;
            transform: translateY(20px);
            animation: slideUpFade 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        @keyframes slideUpFade {{
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .loader-ring {{
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 3px solid rgba(79, 70, 229, 0.1);
            border-radius: 50%;
            border-top-color: #4f46e5;
            animation: spin 1s ease-in-out infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        audio::-webkit-media-controls-panel {{ background-color: transparent; }}
        textarea:focus, select:focus {{ outline: none; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2); border-color: #4f46e5; }}
    </style>
</head>
<body class="text-slate-800 p-4 md:p-8 lg:p-12 relative overflow-x-hidden">

    <div id="loader" class="hidden fixed inset-0 z-50 bg-slate-50/90 backdrop-blur-md flex-col items-center justify-center transition-opacity duration-300">
        <div class="loader-ring mb-6"></div>
        <h2 id="loader-text" class="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-violet-600 tracking-tight">Initializing Synthesis...</h2>
        <p class="text-slate-500 mt-3 text-sm font-medium animate-pulse">Processing natural language and rendering media</p>
    </div>

    <div class="max-w-6xl mx-auto relative z-10">
        <header class="text-center mb-16 mt-6 animate-up" style="animation-delay: 0.1s;">
            <div class="inline-flex items-center gap-2 mb-4 px-4 py-1.5 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 font-semibold text-xs tracking-widest uppercase">
                <span class="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span> Production Build
            </div>
            <h1 class="text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-slate-800 to-indigo-900 tracking-tight mb-6">
                Empathy & Vision Engine
            </h1>
            <p class="text-lg md:text-xl text-slate-500 font-light max-w-2xl mx-auto leading-relaxed">
                Transform unstructured text into emotionally intelligent audio and cinematic visual storyboards instantly.
            </p>
        </header>
        
        <form action="/generate" method="post" onsubmit="showLoadingState()" class="glass-panel p-6 md:p-10 rounded-3xl mb-16 animate-up" style="animation-delay: 0.2s;">
            <div class="mb-8">
                <label class="block text-sm font-bold text-slate-700 mb-3 tracking-wider uppercase">Source Narrative</label>
                <textarea name="user_text" rows="4" required class="w-full p-5 bg-white/80 border border-slate-200 rounded-2xl transition-all resize-none text-lg text-slate-700 placeholder-slate-400" placeholder="Enter your script or story here..."></textarea>
            </div>
            
            <div class="mb-10">
                <label class="block text-sm font-bold text-slate-700 mb-3 tracking-wider uppercase">Visual Art Direction</label>
                <select name="style" class="w-full p-4 bg-white/80 border border-slate-200 rounded-2xl transition-all appearance-none cursor-pointer text-slate-700 font-medium text-lg">
                    <option value="Cinematic photography, photorealistic, 8k resolution, dramatic volumetric lighting, highly detailed">Cinematic & Photorealistic</option>
                    <option value="Flat 2D vector illustration graphic design, solid pastel colors, minimal clean UI style, NO shading, NO 3D elements, pure white background">Corporate Minimalist Vector (2D Flat)</option>
                    <option value="3D isometric render, orthographic projection, soft studio lighting, clean minimal clay render style, pastel colors, tilt-shift lens">3D Isometric Studio</option>
                </select>
            </div>
            
            <button type="submit" class="w-full bg-slate-900 text-white font-semibold text-lg py-4 px-6 rounded-2xl hover:bg-indigo-600 shadow-lg hover:shadow-indigo-500/30 hover:-translate-y-0.5 transition-all duration-300 flex justify-center items-center gap-3">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                Execute Pipeline
            </button>
        </form>

        {results_html}
    </div>

    <script>
        function showLoadingState() {{
            const loader = document.getElementById('loader');
            loader.classList.remove('hidden');
            loader.classList.add('flex');
            
            const messages = [
                "Analyzing semantic emotion...", 
                "Modulating vocal parameters...", 
                "Segmenting narrative arcs...", 
                "Engineering optimal prompts...",
                "Awaiting API response..."
            ];
            let step = 0;
            const textNode = document.getElementById('loader-text');
            setInterval(() => {{
                step = (step + 1) % messages.length;
                textNode.style.opacity = 0;
                setTimeout(() => {{
                    textNode.innerText = messages[step];
                    textNode.style.opacity = 1;
                }}, 200);
            }}, 2500);
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
    <div class="glass-panel p-6 md:p-8 rounded-3xl mb-12 border-l-4 border-indigo-500 animate-up" style="animation-delay: 0.1s;">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
            <h2 class="text-2xl font-bold text-slate-800 tracking-tight flex items-center gap-3">
                <span class="p-2 bg-indigo-100 text-indigo-600 rounded-xl">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>
                </span>
                Vocal Synthesis
            </h2>
            <div class="flex items-center gap-2 bg-white/60 px-4 py-2 rounded-2xl border border-slate-100">
                <span class="text-xs font-bold text-slate-500 uppercase tracking-widest">Emotion:</span>
                <span class="text-indigo-600 font-bold">
                    {audio_data['emotion']} ({audio_data['confidence']}%)
                </span>
            </div>
        </div>
        <div class="bg-white/40 p-2 rounded-2xl border border-white/50 shadow-sm">
            <audio controls class="w-full h-12 outline-none"><source src="{audio_data['audio_path']}" type="audio/mpeg"></audio>
        </div>
    </div>
    """
    
    panels_html = ""
    for idx, panel in enumerate(storyboard_data):
        img_tag = f'<img src="{panel["image_url"]}" alt="Panel {idx+1}" class="w-full h-full object-cover transform hover:scale-105 transition-transform duration-700">'
        panels_html += f"""
        <div class="glass-panel rounded-3xl overflow-hidden flex flex-col hover:-translate-y-2 hover:shadow-2xl transition-all duration-300 animate-up group" style="animation-delay: {0.2 + (idx * 0.1)}s;">
            <div class="relative overflow-hidden h-56 bg-slate-100">
                {img_tag}
                <div class="absolute top-4 left-4 bg-white/90 backdrop-blur text-slate-800 text-xs font-bold px-3 py-1.5 rounded-xl shadow-sm tracking-wide uppercase">Scene {idx + 1}</div>
            </div>
            <div class="p-6 flex-grow flex flex-col bg-white/30">
                <p class="text-slate-800 font-medium text-base mb-6 leading-relaxed flex-grow">"{panel['scene_text']}"</p>
                <div class="mt-auto border-t border-slate-200/50 pt-4">
                    <p class="text-[10px] text-slate-400 font-mono overflow-hidden text-ellipsis line-clamp-2" title="{panel['engineered_prompt']}">
                        <span class="font-bold text-indigo-400">PROMPT:</span> {panel['engineered_prompt']}
                    </p>
                </div>
            </div>
        </div>
        """
        
    storyboard_html = f'<div class="mb-12"><h2 class="text-2xl font-bold text-slate-800 mb-8 tracking-tight animate-up flex items-center gap-3"><span class="p-2 bg-indigo-100 text-indigo-600 rounded-xl"><svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg></span> Visual Storyboard</h2><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">{panels_html}</div></div>'
    
    return HTML_TEMPLATE.format(results_html=audio_html + storyboard_html)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(os.path.join(OUTPUT_DIR, filename))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)