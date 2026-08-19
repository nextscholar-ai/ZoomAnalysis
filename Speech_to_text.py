import os
import time
import json
import subprocess
import sys
from collections import defaultdict
import warnings
import re
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Fix for Python 3.13 import issue
def fix_imports():
    """Try to fix import paths"""
    try:
        import site
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.insert(0, user_site)
            print(f"✓ Added user site-packages: {user_site}")
    except:
        pass
    
    possible_paths = [
        os.path.expanduser("~/.local/lib/python3.13/site-packages"),
        os.path.expanduser("~/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/site-packages")
    ]
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"✓ Added path: {path}")

# Try to fix imports first
fix_imports()

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    print("✓ faster-whisper imported successfully")
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"\n❌ faster-whisper not installed!")
    print("Install with: pip install faster-whisper")
    sys.exit(1)

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=False)
        return True
    except FileNotFoundError:
        return False

def convert_to_wav(audio_file):
    """Convert audio to WAV format using ffmpeg"""
    if not check_ffmpeg():
        print("\n⚠️ FFmpeg not found!")
        print("   Please install FFmpeg from: https://ffmpeg.org/download.html")
        return None

    base_name = os.path.splitext(audio_file)[0]
    output_file = f"{base_name}.wav"

    if os.path.exists(output_file):
        print(f"\n✓ Using existing WAV file: {output_file}")
        return output_file

    print(f"\n🔄 Converting audio to WAV format...")
    try:
        subprocess.run([
            'ffmpeg', '-i', audio_file, 
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_file
        ], check=True, capture_output=True, text=True)
        print(f"   ✓ Converted to: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Conversion failed: {e.stderr}")
        return None

def time_to_seconds(t):
    """Convert time string to seconds"""
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        return 0

def load_zoom_timeline(json_file):
    """Load Zoom timeline from JSON"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("timeline", [])
    except FileNotFoundError:
        print(f"❌ JSON file not found: {json_file}")
        return []
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {json_file}")
        return []

def get_speaker_intervals(timeline):
    """Extract speaker intervals from timeline"""
    speaker_intervals = defaultdict(list)
    current_speakers = set()
    last_time = None

    for entry in timeline:
        current_time = time_to_seconds(entry.get("ts", "0:0:0"))
        users = entry.get("users", [])

        current_users = set()
        for user in users:
            username = user.get("username", "").strip()
            if username:
                zoom_userid = user.get("zoom_userid", "").strip()
                if zoom_userid:
                    username = "TEACHER"
                current_users.add(username)

        if last_time is not None:
            stopped_speakers = current_speakers - current_users
            for speaker in stopped_speakers:
                if speaker_intervals[speaker]:
                    speaker_intervals[speaker][-1]['end'] = last_time

            started_speakers = current_users - current_speakers
            for speaker in started_speakers:
                speaker_intervals[speaker].append({
                    'start': last_time,
                    'end': None
                })

        current_speakers = current_users
        last_time = current_time

    for speaker, intervals in speaker_intervals.items():
        for interval in intervals:
            if interval['end'] is None:
                interval['end'] = last_time

    return speaker_intervals

def detect_hallucination(text):
    """
    Advanced hallucination detection
    
    Returns: (is_hallucination, reason)
    """
    text_clean = text.strip().lower()
    
    if not text_clean or len(text_clean) < 5:
        return True, "too_short"
    
    words = text_clean.split()
    
    # 1. Check for excessive repetition of same words
    word_counts = Counter(words)
    max_repeat = max(word_counts.values()) if word_counts else 0
    
    if max_repeat > len(words) * 0.5 and len(words) > 5:
        return True, f"excessive_repetition_{max_repeat}_{len(words)}"
    
    # 2. Check for common hallucination patterns
    hallucination_patterns = [
        (r'(सब्सक्राइब\s*करें\s*और\s*){3,}', 'subscribe_repeat'),  # 3+ times
        (r'(लाइक\s*करें\s*और\s*){3,}', 'like_repeat'),
        (r'(शेयर\s*करें\s*और\s*){3,}', 'share_repeat'),
        (r'(धन्यवाद\s*){3,}', 'thanks_repeat'),
        (r'(नमस्ते\s*){3,}', 'namaste_repeat'),
        (r'(हाँ\s*){5,}', 'yes_repeat'),
        (r'(नहीं\s*){5,}', 'no_repeat'),
        (r'(वालेकुम\s*सर\s*){3,}', 'walaikum_repeat'),
        (r'(असलाम\s*वालेकुम\s*){3,}', 'asalam_repeat'),
        (r'(ok\s*){5,}', 'ok_repeat'),
        (r'(hmm\s*){5,}', 'hmm_repeat'),
    ]
    
    for pattern, reason in hallucination_patterns:
        if re.search(pattern, text_clean):
            return True, reason
    
    # 3. Check if text consists mostly of same word with small variations
    unique_words = set(words)
    if len(unique_words) <= 2 and len(words) > 10:
        return True, "too_few_unique_words"
    
    # 4. Check for very short repetitive segments
    if len(text_clean) < 10 and max_repeat > 2:
        return True, "short_repetitive"
    
    return False, "clean"

def transcribe_with_faster_whisper_large(audio_file):
    """Transcribe with hallucination prevention"""
    if not WHISPER_AVAILABLE:
        print("\n❌ faster-whisper not available.")
        return None, None

    print("\n" + "="*80)
    print(" 🎙️ WHISPER TRANSCRIPTION (HALLUCINATION PREVENTION)")
    print("="*80)

    if not os.path.exists(audio_file):
        print(f"\n❌ Audio file not found: {audio_file}")
        return None, None

    # Convert to WAV if needed
    if audio_file.lower().endswith(('.m4a', '.mp3', '.aac', '.flac')):
        wav_file = convert_to_wav(audio_file)
        if wav_file:
            audio_file = wav_file
        else:
            return None, None

    file_size = os.path.getsize(audio_file) / (1024 * 1024)
    print(f"\n📁 Audio: {os.path.basename(audio_file)} ({file_size:.1f} MB)")

    model_size = "large-v3"
    compute_type = "int8"
    
    print(f"\n🚀 Loading faster-whisper model: {model_size}")
    print(f"   Compute type: {compute_type}")
    print(f"   Device: CPU")
    print(f"   First download may take 2-3 minutes (model size: ~3GB)...")
    
    start_load = time.time()
    
    try:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=8,
            num_workers=1
        )
        
        load_time = time.time() - start_load
        print(f"   ✓ Model loaded in {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return None, None

    print(f"\n🎤 Transcribing audio with Whisper large-v3...")
    print(f"   ⏳ Processing {file_size:.1f} MB file...")
    print()

    start_time = time.time()

    try:
        # CRITICAL FIX: Use temperature fallback to prevent hallucinations
        segments, info = model.transcribe(
            audio_file,
            language="hi",
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Temperature fallback
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 1200,  # Much higher
                "threshold": 0.7,
                "min_speech_duration_ms": 800,  # Much higher
            },
            condition_on_previous_text=True,
            no_speech_threshold=0.8,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-0.5,
            word_timestamps=False,
        )
        
        all_segments = []
        segment_count = 0
        hallucination_count = 0
        last_percent = -1
        
        print("📝 Transcription in progress (filtering hallucinations):")
        print("-" * 80)
        
        for segment in segments:
            text = segment.text.strip()
            duration = segment.end - segment.start
            
            # Skip too short or too long segments
            if duration < 1.0 or duration > 30.0:
                continue
            
            # Check for hallucination
            is_hall, reason = detect_hallucination(text)
            
            if is_hall:
                hallucination_count += 1
                continue
            
            if text and len(text) > 5:
                all_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text
                })
                segment_count += 1
                
                try:
                    percent = int((segment.end / info.duration) * 100)
                    if percent != last_percent and percent % 10 == 0:
                        print(f"\r   Progress: {percent}% ({segment_count} valid segments, {hallucination_count} hallucinations filtered)", end="", flush=True)
                        last_percent = percent
                except:
                    pass
                
                if segment_count <= 5:
                    print(f"\n[{segment.start:.0f}s -> {segment.end:.0f}s] {text[:150]}...")
        
        print(f"\r   Progress: 100% ({segment_count} valid segments, {hallucination_count} hallucinations filtered)    ")
        
        elapsed = time.time() - start_time
        
        print("-" * 80)
        print(f"\n✅ Transcription completed in {elapsed:.1f} seconds")
        print(f"   Valid segments: {len(all_segments)}")
        print(f"   Hallucinations filtered: {hallucination_count}")
        try:
            print(f"   Duration: {info.duration:.0f} seconds ({info.duration/60:.1f} minutes)")
            print(f"   Language: {info.language} ({info.language_probability:.0%} confidence)")
            print(f"   Speed: {info.duration/elapsed:.1f}x real-time")
        except:
            pass
        
        class Info:
            def __init__(self, duration):
                self.duration = duration
                self.language = "hi"
                self.language_probability = 0.95
        
        info_obj = Info(info.duration if hasattr(info, 'duration') else 0)
        
        return all_segments, info_obj
        
    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def match_with_speakers(transcript_segments, timeline):
    """Match transcription with speakers"""
    print("\n🔍 Matching segments with speakers...")

    if not timeline:
        speaker_segments = defaultdict(list)
        for segment in transcript_segments:
            segment['speaker'] = "Unknown"
            speaker_segments["Unknown"].append(segment)
        return transcript_segments, speaker_segments

    speaker_intervals = get_speaker_intervals(timeline)

    if not speaker_intervals:
        speaker_segments = defaultdict(list)
        for segment in transcript_segments:
            segment['speaker'] = "Unknown"
            speaker_segments["Unknown"].append(segment)
        return transcript_segments, speaker_segments

    # Build speaker lookup with tolerance
    speaker_at_time = {}
    for speaker, intervals in speaker_intervals.items():
        for interval in intervals:
            start = int(interval['start'])
            end = int(interval['end'])
            for t in range(max(0, start - 3), end + 3):
                speaker_at_time[t] = speaker

    matched_segments = []
    speaker_segments = defaultdict(list)

    for segment in transcript_segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        # Check multiple time points
        mid_time = int((start_time + end_time) / 2)
        speaker = speaker_at_time.get(mid_time)
        
        if speaker is None:
            speaker = speaker_at_time.get(int(start_time))
        if speaker is None:
            speaker = speaker_at_time.get(int(end_time))
        if speaker is None:
            # Check around the segment
            for t in range(int(start_time), int(end_time) + 1, 10):
                if t in speaker_at_time:
                    speaker = speaker_at_time[t]
                    break
        if speaker is None:
            speaker = "Unknown"

        matched_segment = {
            'start': start_time,
            'end': end_time,
            'text': text,
            'speaker': speaker
        }

        matched_segments.append(matched_segment)
        speaker_segments[speaker].append(matched_segment)

    print(f"   ✓ Matched with {len(speaker_segments)} speakers")
    return matched_segments, speaker_segments

def process_and_clean_segments(matched_segments):
    """Apply all cleaning filters to segments"""
    print("\n🧹 Final cleaning...")
    
    original_count = len(matched_segments)
    
    # 1. Remove any remaining hallucinations
    cleaned = []
    for seg in matched_segments:
        is_hall, reason = detect_hallucination(seg['text'])
        if not is_hall:
            cleaned.append(seg)
    
    # 2. Merge adjacent segments from same speaker (if gap < 2s)
    merged = []
    if cleaned:
        current = cleaned[0].copy()
        for seg in cleaned[1:]:
            gap = seg['start'] - current['end']
            if (current['speaker'] == seg['speaker'] and 
                gap < 2.0 and 
                len(current['text']) + len(seg['text']) < 300):
                current['end'] = seg['end']
                current['text'] = current['text'] + " " + seg['text']
            else:
                merged.append(current)
                current = seg.copy()
        merged.append(current)
    
    # 3. Remove very short segments
    final = [s for s in merged if s['end'] - s['start'] >= 1.5]
    
    print(f"   📊 Original: {original_count} segments")
    print(f"   📊 After cleaning: {len(final)} segments")
    print(f"   📊 Removed: {original_count - len(final)} segments")
    
    return final

def save_transcription(matched_segments, audio_file, info):
    """Save transcription with speakers"""
    output_dir = "transcriptions"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_transcript.txt")

    # Group by speaker
    speaker_segments = defaultdict(list)
    for seg in matched_segments:
        speaker_segments[seg['speaker']].append(seg)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" 🎙️ CLEAN TRANSCRIPTION (HALLUCINATIONS REMOVED)\n")
        f.write("="*80 + "\n")
        f.write(f" Audio: {audio_file}\n")
        try:
            f.write(f" Duration: {info.duration:.0f}s ({info.duration/60:.1f} min)\n")
        except:
            pass
        f.write(f" Total Segments: {len(matched_segments)}\n")
        f.write("="*80 + "\n\n")

        for speaker, segments in speaker_segments.items():
            if not segments:
                continue
            
            # Double-check filtering
            valid_segments = []
            for seg in segments:
                is_hall, _ = detect_hallucination(seg['text'])
                if not is_hall:
                    valid_segments.append(seg)
            
            if not valid_segments:
                continue
                
            f.write(f"\n{'='*80}\n")
            f.write(f"👤 {speaker} ({len(valid_segments)} segments)\n")
            f.write(f"{'='*80}\n\n")

            for segment in valid_segments:
                start = segment['start']
                end = segment['end']
                text = segment['text']
                f.write(f"[{start:.0f}s -> {end:.0f}s]\n")
                f.write(f"   {text}\n\n")

    print(f"💾 Transcript saved to: {output_file}")

    # Save cleaned JSON
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'file': audio_file,
            'duration': info.duration if info else 0,
            'segments': matched_segments,
            'speaker_summary': {
                speaker: len(segments) 
                for speaker, segments in speaker_segments.items() if segments
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 JSON saved to: {json_file}")
    
    return output_file

def main():
    """Main execution"""
    AUDIO_FILE = "NS26A_ Math - Bhawna Mam_20260220_155842_audio_only.m4a"
    JSON_FILE = "NS26A_ Math - Bhawna Mam_20260220_155842_timeline.json"

    print("\n" + "="*80)
    print(" 🎯 WHISPER TRANSCRIPTION - HALLUCINATION FIXED")
    print("="*80)

    if not os.path.exists(AUDIO_FILE):
        print(f"\n❌ Audio file not found: {AUDIO_FILE}")
        print("\n💡 Please update the AUDIO_FILE variable with your file name")
        return

    print(f"\n📂 Loading Zoom timeline...")
    timeline = load_zoom_timeline(JSON_FILE)
    if timeline:
        print(f"   ✓ Loaded {len(timeline)} entries")
    else:
        print(f"   ⚠️ Timeline file not found or empty")

    print("\n" + "="*80)
    print(" 🎯 KEY FIXES APPLIED")
    print("="*80)
    print("   ✅ Temperature fallback (0.0 to 1.0)")
    print("   ✅ Higher VAD thresholds")
    print("   ✅ Hallucination detection patterns")
    print("   ✅ Aggressive filtering")
    print("="*80)

    # Transcribe
    transcript_segments, info = transcribe_with_faster_whisper_large(AUDIO_FILE)

    if not transcript_segments:
        print("\n❌ No valid transcription segments found")
        print("\n💡 If you're still seeing hallucinations, try:")
        print("   1. Convert audio to WAV manually")
        print("   2. Use a smaller model: model_size = 'medium'")
        print("   3. Check if the audio file has background music")
        return

    # Match with speakers
    matched_segments, _ = match_with_speakers(transcript_segments, timeline)

    # Clean
    cleaned_segments = process_and_clean_segments(matched_segments)

    # Show sample
    print("\n" + "="*80)
    print(" 📝 SAMPLE CLEAN TRANSCRIPTION")
    print("="*80)
    print()

    for i, segment in enumerate(cleaned_segments[:10]):
        speaker = segment.get('speaker', 'Unknown')
        text = segment['text']
        start = segment['start']
        end = segment['end']
        emoji = "👨🏫" if speaker == "TEACHER" else "👤"
        print(f"[{start:.0f}s -> {end:.0f}s] {emoji} {speaker}:")
        print(f"   {text}")
        print()

    # Save
    save_transcription(cleaned_segments, AUDIO_FILE, info)

    print("\n" + "="*80)
    print(" ✅ COMPLETE!")
    print("   📁 Check the 'transcriptions' folder for clean output")
    print("="*80)

if __name__ == "__main__":
    main()