"""
Simple Hindi to English Translation with Timeline
Shows the conversation chronologically with translations
"""

import os
import json
import re
from collections import defaultdict
import ollama
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "richardyoung/llama-3.2-3b-instruct-abliterated:Q4_K_M"
INPUT_JSON = "NS26A_ Math - Bhawna Mam_20260220_155842_audio_only_with_speakers.json"
OUTPUT_FILE = "translated_conversation.txt"

# ==========================================
# LOAD AND PROCESS
# ==========================================

def load_transcript(json_file):
    """Load the transcript JSON file"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'segments' in data:
            return data['segments']
        else:
            return []
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

def is_noise_text(text):
    """Check if text is noise/hallucination"""
    noise_patterns = [
        'subscribe', 'सब्सक्राइब',
        'like', 'लाइक',
        'share', 'शेयर',
        '...',
        'music', 'applause',
        'speech recognition',
    ]
    
    text_lower = text.lower().strip()
    
    for pattern in noise_patterns:
        if pattern.lower() in text_lower:
            return True
    
    if len(text) < 5:
        return True
    
    return False

def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def translate_to_english(hindi_text):
    """Translate Hindi to English using Llama"""
    if not hindi_text or len(hindi_text) < 3:
        return None
    
    try:
        prompt = f"""Translate this Hindi text to natural English. Only provide the translation, no explanations.

Hindi: {hindi_text}

English translation:"""
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': 'You are a translator. Translate Hindi to natural English.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        translation = response['message']['content'].strip()
        # Clean up any extra text
        translation = re.sub(r'^English translation:?\s*', '', translation, flags=re.IGNORECASE)
        translation = re.sub(r'^Translation:?\s*', '', translation, flags=re.IGNORECASE)
        translation = re.sub(r'^"|"$', '', translation)
        
        return translation if translation else None
        
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return None

def format_time(seconds):
    """Format seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ==========================================
# MAIN PROCESSING
# ==========================================

def main():
    print("\n" + "=" * 100)
    print(" " * 30 + "HINDI TO ENGLISH TRANSLATION (TIMELINE)")
    print("=" * 100)
    
    # Check Ollama
    try:
        ollama.list()
        print("✅ Ollama is running")
        print(f"✅ Using model: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        return
    
    # Load transcript
    print(f"\n📂 Loading: {INPUT_JSON}")
    segments = load_transcript(INPUT_JSON)
    
    if not segments:
        print("❌ No data found")
        return
    
    print(f"   ✓ Loaded {len(segments)} segments")
    
    # Filter and sort by time
    print("\n🔍 Processing and translating...")
    
    translated_segments = []
    noise_count = 0
    
    for seg in segments:
        text = seg.get('text', '').strip()
        speaker = seg.get('speaker', 'Unknown')
        start_time = seg.get('start', 0)
        end_time = seg.get('end', 0)
        
        if not text:
            continue
        
        # Clean text
        cleaned = clean_text(text)
        if not cleaned or is_noise_text(cleaned):
            noise_count += 1
            continue
        
        # Translate
        english = translate_to_english(cleaned)
        if not english:
            continue
        
        translated_segments.append({
            'speaker': speaker,
            'hindi': cleaned,
            'english': english,
            'start': start_time,
            'end': end_time,
            'time': format_time(start_time)
        })
    
    # Sort by start time
    translated_segments.sort(key=lambda x: x['start'])
    
    print(f"   ✓ Translated {len(translated_segments)} segments")
    print(f"   ✓ Filtered out {noise_count} noise segments")
    
    if not translated_segments:
        print("❌ No valid segments to translate")
        return
    
    # Show sample
    print("\n" + "=" * 100)
    print(" 📝 TRANSLATION PREVIEW (First 5 segments)")
    print("=" * 100)
    print()
    
    for i, seg in enumerate(translated_segments[:5]):
        print(f"[{seg['time']}] {seg['speaker']}:")
        print(f"   Hindi: {seg['hindi'][:100]}...")
        print(f"   English: {seg['english'][:100]}...")
        print()
    
    # Save to file
    print("\n💾 Saving translations...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write(" " * 35 + "HINDI TO ENGLISH TRANSLATION (TIMELINE)\n")
        f.write("=" * 120 + "\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" Source: {INPUT_JSON}\n")
        f.write(f" Total Segments: {len(translated_segments)}\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("📜 CONVERSATION TIMELINE\n")
        f.write("-" * 120 + "\n\n")
        
        for seg in translated_segments:
            f.write(f"[{seg['time']}] 👤 {seg['speaker']}:\n")
            f.write(f"   📝 Hindi: {seg['hindi']}\n")
            f.write(f"   🇬🇧 English: {seg['english']}\n")
            f.write("\n" + "─" * 80 + "\n\n")
        
        # Add summary
        f.write("\n" + "=" * 120 + "\n")
        f.write(" 📊 SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        
        # Count by speaker
        speaker_counts = defaultdict(int)
        for seg in translated_segments:
            speaker_counts[seg['speaker']] += 1
        
        f.write("👥 Speakers:\n")
        for speaker, count in speaker_counts.items():
            f.write(f"   • {speaker}: {count} segments\n")
        
        # Count total words
        total_words = sum(len(seg['english'].split()) for seg in translated_segments)
        f.write(f"\n📊 Total Words (English): {total_words}\n")
        f.write(f"📊 Total Segments: {len(translated_segments)}\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write(" " * 45 + "END OF TRANSCRIPT\n")
        f.write("=" * 120 + "\n")
    
    print(f"✅ Saved to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 100)
    print(" " * 35 + "TRANSLATION COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()