import os
import json
import re
from collections import defaultdict
import ollama
from datetime import datetime



MODEL_NAME = "richardyoung/llama-3.2-3b-instruct-abliterated:Q4_K_M"
INPUT_JSON = "audio_file_with_speakers.json"
OUTPUT_TXT = "translated_conversation.txt"
OUTPUT_JSON = "translated_conversation.json"
OUTPUT_ENGLISH_ONLY_JSON = "english_only_transcript.json" 



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

def is_disturbance_text(text):
    """Check if text is disturbance/noise/placeholder"""
    disturbance_patterns = [
        'Hindi and English mixed speech conversation',
        'Hindi and English mixed',
        'speech conversation',
        'subtitle',
        'caption',
        'transcription',
        'speech recognition',
        '[Music]',
        '[Applause]',
        '[Silence]',
        '...',
        '. . .',
        '  ',
        'Note:',
        'Translation:',
        'However, without more context',
        'If you could provide',
        'None of these words appear',
    ]
    
    text_clean = text.strip()
    
    for pattern in disturbance_patterns:
        if pattern.lower() in text_clean.lower():
            return True
    
    if len(text_clean) < 4:
        return True
    
    if re.match(r'^[\s\d\W]+$', text_clean):
        return True
    
    return False

def is_valid_hindi_text(text):
    """Check if text contains actual Hindi characters"""
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    hindi_chars = hindi_pattern.findall(text)
    
    # Check if text has meaningful Hindi content
    if len(hindi_chars) < 2:
        return False
    
    # Check if text has actual words (not just random characters)
    words = text.split()
    if len(words) < 1:
        return False
    
    return True

def clean_text(text):
    """Clean and normalize text"""
    # Remove brackets and parentheses
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_subject_related(text):
    """Check if text is related to the subject (Math/Science)"""
    # Keywords related to Math and Science
    math_keywords = [
        'math', 'science', 'number', 'prime', 'angle', 'degree', 'equation',
        'calculate', 'solve', 'formula', 'geometry', 'algebra', 'graph',
        'ratio', 'percentage', 'fraction', 'decimal', 'square', 'root',
        'sum', 'difference', 'product', 'divide', 'multiply', 'add', 'subtract',
        'triangle', 'circle', 'line', 'point', 'parallel', 'perpendicular',
        'area', 'volume', 'perimeter', 'radius', 'diameter', 'pi',
        'learn', 'understand', 'concept', 'example', 'question', 'answer',
        'समझ', 'सवाल', 'जवाब', 'गणित', 'विज्ञान', 'संख्या', 'कोण'
    ]
    
    text_lower = text.lower()
    
    # Check if any keyword is present
    for keyword in math_keywords:
        if keyword in text_lower:
            return True
    
    # Check if it's a question (has question mark or question words)
    question_patterns = [
        r'[?？]',
        r'\b(?:what|why|how|when|where|who|which|is|are|was|were|do|does|did|has|have|will|shall|can|could|would|should|may|might|must)\b'
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False

def translate_to_english(hindi_text, context=None):
    
    if not hindi_text or len(hindi_text) < 3:
        return None
    
    try:
        
        prompt = f"""You are a translator for an educational classroom. Translate this Hindi text to natural, clear English.

IMPORTANT RULES:
1. ONLY provide the English translation - no explanations, no notes
2. Make the translation meaningful and grammatically correct
3. If it's about Math/Science, use proper subject terminology
4. Keep the tone natural, like a classroom conversation
5. Do not translate: mathematical symbols, numbers, or formulas (keep them as is)
6. If the text is not a complete sentence, make it understandable

Hindi: {hindi_text}

English translation:"""
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': 'You are an educational translator. Translate Hindi to natural, clear English. Only provide the translation.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        translation = response['message']['content'].strip()
        
        # Clean up the translation
        translation = re.sub(r'^English translation:?\s*', '', translation, flags=re.IGNORECASE)
        translation = re.sub(r'^Translation:?\s*', '', translation, flags=re.IGNORECASE)
        translation = re.sub(r'^"|"$', '', translation)
        
        # Remove any extra notes or explanations
        if 'Note:' in translation:
            translation = translation.split('Note:')[0].strip()
        if 'Note -' in translation:
            translation = translation.split('Note -')[0].strip()
        
        # If translation is empty or too short, return None
        if not translation or len(translation) < 3:
            return None
        
       
        if hindi_text.strip().endswith('?'):
            if not translation.strip().endswith('?'):
                translation = translation.strip() + '?'
        
        return translation
        
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return None

def format_time(seconds):
    """Format seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"



def main():
    print("\n" + "=" * 120)
    print(" " * 35 + "HINDI TO ENGLISH TRANSLATION (Enhanced)")
    print("=" * 120)
    
    
    try:
        ollama.list()
        print("✅ Ollama is running")
        print(f"✅ Using model: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        return
    
    
    print(f"\n📂 Loading: {INPUT_JSON}")
    segments = load_transcript(INPUT_JSON)
    
    if not segments:
        print("❌ No data found")
        return
    
    print(f"   ✓ Loaded {len(segments)} segments")
    
    
    print("\n🔍 Filtering, analyzing subject context, and translating...")
    
    translated = []
    filtered_count = 0
    disturbance_count = 0
    non_subject_count = 0
    translation_errors = 0
    
    for seg in segments:
        text = seg.get('text', '').strip()
        speaker = seg.get('speaker', 'Unknown')
        start_time = seg.get('start', 0)
        end_time = seg.get('end', 0)
        
        if not text:
            continue
        
        
        cleaned = clean_text(text)
        
        
        if is_disturbance_text(cleaned):
            disturbance_count += 1
            continue
        
        
        if not is_valid_hindi_text(cleaned):
            filtered_count += 1
            continue
        
        
        if len(cleaned) < 4:
            filtered_count += 1
            continue
        
        
        if not is_subject_related(cleaned):
            non_subject_count += 1
            continue
        
        
        english = translate_to_english(cleaned)
        if not english:
            translation_errors += 1
            continue
        
        translated.append({
            'time': format_time(start_time),
            'time_seconds': start_time,
            'speaker': speaker,
            'hindi': cleaned,
            'english': english
        })
    
   
    translated.sort(key=lambda x: x['time_seconds'])
    
    print(f"\n   ✓ Translated: {len(translated)} sentences")
    print(f"   ✓ Filtered disturbance: {disturbance_count}")
    print(f"   ✓ Filtered invalid text: {filtered_count}")
    print(f"   ✓ Filtered non-subject: {non_subject_count}")
    print(f"   ✓ Translation errors: {translation_errors}")
    
    if not translated:
        print("❌ No valid sentences to translate")
        return
    
    # Show preview
    print("\n" + "=" * 120)
    print(" 📝 PREVIEW (First 8 translations)")
    print("=" * 120)
    print()
    
    for i, item in enumerate(translated[:8]):
        print(f"[{item['time']}] {item['speaker']}:")
        print(f"   Hindi: {item['hindi'][:100]}...")
        print(f"   English: {item['english'][:100]}...")
        print()
    
    print("\n💾 Saving TXT file...")
    
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write(" " * 35 + "HINDI TO ENGLISH TRANSLATION\n")
        f.write("=" * 120 + "\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" Source: {INPUT_JSON}\n")
        f.write(f" Model: {MODEL_NAME}\n")
        f.write(f" Total Sentences: {len(translated)}\n")
        f.write("=" * 120 + "\n\n")
        
        for item in translated:
            f.write(f"[{item['time']}] {item['speaker']}:\n")
            f.write(f"   Hindi: {item['hindi']}\n")
            f.write(f"   English: {item['english']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"   ✅ TXT saved: {OUTPUT_TXT}")
    
    
    print("\n💾 Saving JSON file (with metadata)...")
    
    json_output = {
        "metadata": {
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source_file": INPUT_JSON,
            "model": MODEL_NAME,
            "total_sentences": len(translated),
            "filtered_disturbance": disturbance_count,
            "filtered_invalid": filtered_count,
            "filtered_non_subject": non_subject_count,
            "translation_errors": translation_errors
        },
        "translations": translated
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ JSON saved: {OUTPUT_JSON}")
    

    print("\n💾 Saving English-only JSON file...")
    
    english_only = []
    for item in translated:
        english_only.append({
            'time': item['time'],
            'time_seconds': item['time_seconds'],
            'speaker': item['speaker'],
            'english': item['english']
        })
    
    with open(OUTPUT_ENGLISH_ONLY_JSON, 'w', encoding='utf-8') as f:
        json.dump(english_only, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ English-only JSON saved: {OUTPUT_ENGLISH_ONLY_JSON}")
    
    
    english_only_txt = OUTPUT_ENGLISH_ONLY_JSON.replace('.json', '.txt')
    
    with open(english_only_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write(" " * 35 + "ENGLISH ONLY TRANSCRIPT\n")
        f.write("=" * 120 + "\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" Total Sentences: {len(english_only)}\n")
        f.write("=" * 120 + "\n\n")
        
        for item in english_only:
            f.write(f"[{item['time']}] {item['speaker']}:\n")
            f.write(f"   {item['english']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"   ✅ English-only TXT saved: {english_only_txt}")
    
    
    print("\n" + "=" * 120)
    print(" 📊 TRANSLATION STATISTICS")
    print("=" * 120)
    print()
    
    speakers = defaultdict(int)
    for item in translated:
        speakers[item['speaker']] += 1
    
    print(" SPEAKER BREAKDOWN:")
    for speaker, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True):
        print(f"   {speaker}: {count} sentences")
    
    print(f"\n OUTPUT FILES:")
    print(f"   Full translation (TXT): {OUTPUT_TXT}")
    print(f"   Full translation (JSON): {OUTPUT_JSON}")
    print(f"   English only (JSON): {OUTPUT_ENGLISH_ONLY_JSON}")
    print(f"   English only (TXT): {english_only_txt}")
    
    print("\n" + "=" * 120)
    print(" " * 35 + "TRANSLATION COMPLETE")
    print("=" * 120)

if __name__ == "__main__":
    main()
