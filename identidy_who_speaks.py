import os
import time
import json
from faster_whisper import WhisperModel
from collections import defaultdict



def time_to_seconds(t):
    """Convert timestamp string to seconds"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_zoom_timeline(json_file):
    """Load Zoom meeting timeline from JSON file"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["timeline"]

def get_speaker_at_time(timeline, timestamp):
    """
    Find who was speaking at a given timestamp
    Returns: username or None
    """
    for entry in timeline:
        entry_time = time_to_seconds(entry["ts"])
        
        # Check if this timestamp is close to the entry time
        if abs(entry_time - timestamp) < 1.0:  # Within 1 second
            users = entry.get("users", [])
            if users:
                # Return the first user (there could be multiple)
                return users[0].get("username", "Unknown")
    
    # If exact match not found, find the nearest entry
    nearest_time = None
    nearest_user = None
    
    for entry in timeline:
        entry_time = time_to_seconds(entry["ts"])
        users = entry.get("users", [])
        
        if users:
            if nearest_time is None or abs(entry_time - timestamp) < abs(nearest_time - timestamp):
                nearest_time = entry_time
                nearest_user = users[0].get("username", "Unknown")
    
    return nearest_user

def get_speaker_intervals(timeline):
    """
    Convert timeline to speaker intervals
    Returns: dict {username: [(start, end), ...]}
    """
    speaker_intervals = defaultdict(list)
    current_speakers = set()
    last_time = None
    
    for entry in timeline:
        current_time = time_to_seconds(entry["ts"])
        users = entry.get("users", [])
        
        # Get current speakers
        current_users = set()
        for user in users:
            username = user.get("username", "").strip()
            if username:
                current_users.add(username)
        
        # If we have a previous state, close intervals for speakers who stopped
        if last_time is not None:
            # Speakers who stopped
            stopped_speakers = current_speakers - current_users
            for speaker in stopped_speakers:
                if speaker_intervals[speaker]:
                    # Update the last interval's end time
                    speaker_intervals[speaker][-1]['end'] = last_time
            
            # Speakers who started
            started_speakers = current_users - current_speakers
            for speaker in started_speakers:
                speaker_intervals[speaker].append({
                    'start': last_time,
                    'end': None  # Will be updated when they stop
                })
        
        current_speakers = current_users
        last_time = current_time
    
    # Close any remaining open intervals
    for speaker, intervals in speaker_intervals.items():
        for interval in intervals:
            if interval['end'] is None:
                interval['end'] = last_time
    
    return speaker_intervals



def match_transcription_with_speakers(transcript_segments, timeline):
    """
    Match each transcribed segment with the speaker from Zoom timeline
    """
    print("\n" + "="*60)
    print(" 🔍 MATCHING TRANSCRIPTION WITH SPEAKERS")
    print("="*60)
    
    # Get speaker intervals
    speaker_intervals = get_speaker_intervals(timeline)
    
    # Create a mapping for quick speaker lookup
    speaker_at_time = {}
    
    for speaker, intervals in speaker_intervals.items():
        for interval in intervals:
            start = interval['start']
            end = interval['end']
            # Mark each second in the interval
            for t in range(int(start), int(end) + 1):
                speaker_at_time[t] = speaker
    
    # Match each transcription segment
    matched_segments = []
    speaker_segments = defaultdict(list)
    
    print("\n📝 Matching segments with speakers...")
    
    for segment in transcript_segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        # Find who was speaking during this segment
        # Check the middle of the segment
        mid_time = int((start_time + end_time) / 2)
        
        # Look for speaker in a small window around the segment
        speaker = None
        for t in range(int(start_time), int(end_time) + 1):
            if t in speaker_at_time:
                speaker = speaker_at_time[t]
                break
        
        # If not found, try nearby times
        if speaker is None:
            for offset in range(1, 30):
                for t in [int(start_time) - offset, int(end_time) + offset]:
                    if t in speaker_at_time:
                        speaker = speaker_at_time[t]
                        break
                if speaker:
                    break
        
        # If still not found, mark as Unknown
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
    
    return matched_segments, speaker_segments



def transcribe_with_speakers(audio_file, json_file, model_size="small"):
    """
    Transcribe audio and identify who is speaking
    """
    
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    print("\n" + "="*70)
    print(" 🎙️ TRANSCRIPTION WITH SPEAKER IDENTIFICATION")
    print("="*70)
    
    # Check files
    if not os.path.exists(audio_file):
        print(f"\n❌ Audio file not found: {audio_file}")
        return None
    
    if not os.path.exists(json_file):
        print(f"\n❌ JSON file not found: {json_file}")
        print("   Looking for: NS26B_Math-Science Danish Sir_20260220_155725_timeline.json")
        print("   Make sure the file is in the same directory")
        return None
    
    # File info
    file_size = os.path.getsize(audio_file) / (1024 * 1024)
    print(f"\n📁 Audio: {os.path.basename(audio_file)} ({file_size:.1f} MB)")
    print(f"📁 Timeline: {os.path.basename(json_file)}")
    
    # Load Zoom timeline
    print("\n📂 Loading Zoom timeline...")
    timeline = load_zoom_timeline(json_file)
    print(f"   ✓ Loaded {len(timeline)} timeline entries")
    
    # Load Whisper model
    print(f"\n🚀 Loading Whisper model: {model_size}")
    print(f"   This may take 1-2 minutes for first download...")
    
    start_load = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    load_time = time.time() - start_load
    print(f"   ✓ Model loaded in {load_time:.1f} seconds")
    
    # Transcribe audio
    print(f"\n🎤 Transcribing audio...")
    start_time = time.time()
    
    segments, info = model.transcribe(
        audio_file,
        language="hi",
        beam_size=5,
        vad_filter=True,
        initial_prompt="Hindi and English mixed speech conversation."
    )
    
    # Collect transcription segments
    transcript_segments = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': text
            })
    
    elapsed = time.time() - start_time
    
    print(f"\n   ✓ Transcribed {len(transcript_segments)} segments")
    print(f"   ✓ Duration: {info.duration:.0f} seconds ({info.duration/60:.1f} minutes)")
    print(f"   ✓ Time taken: {elapsed:.1f} seconds")
    
    # Match with speakers
    matched_segments, speaker_segments = match_transcription_with_speakers(
        transcript_segments, timeline
    )
    
    # Print results with speaker names
    print("\n" + "="*70)
    print(" 📝 TRANSCRIPTION WITH SPEAKERS")
    print("="*70)
    print()
    
    for segment in matched_segments:
        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        end = segment['end']
        
        # Add emoji for teacher
        if speaker == "Danish Hayat" or speaker == "Danish Hayat Sir":
            speaker_display = f"👨‍🏫 {speaker}"
        else:
            speaker_display = f"👤 {speaker}"
        
        print(f"[{start:.0f}s -> {end:.0f}s] {speaker_display}:")
        print(f"   {text}")
        print()
    
    # Save with speakers
    output_file = audio_file.replace('.m4a', f'_with_speakers.txt').replace('.wav', '_with_speakers.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f" TRANSCRIPTION WITH SPEAKERS\n")
        f.write(f" Audio: {audio_file}\n")
        f.write(f" Model: {model_size}\n")
        f.write("="*70 + "\n\n")
        
        for segment in matched_segments:
            speaker = segment['speaker']
            text = segment['text']
            start = segment['start']
            end = segment['end']
            
            f.write(f"[{start:.0f}s -> {end:.0f}s] {speaker}:\n")
            f.write(f"   {text}\n\n")
    
    print(f"💾 Saved to: {output_file}")
    
    # Save JSON with speakers
    json_file_out = audio_file.replace('.m4a', f'_with_speakers.json').replace('.wav', '_with_speakers.json')
    with open(json_file_out, 'w', encoding='utf-8') as f:
        json.dump({
            'file': audio_file,
            'duration': info.duration,
            'segments': matched_segments,
            'speaker_summary': {
                speaker: len(segments) 
                for speaker, segments in speaker_segments.items()
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON saved to: {json_file_out}")
    
    # Summary by speaker
    print("\n" + "="*70)
    print(" 📊 SPEAKER SUMMARY")
    print("="*70)
    
    total_segments = len(matched_segments)
    for speaker, segments in speaker_segments.items():
        count = len(segments)
        percentage = (count / total_segments) * 100
        print(f"   {speaker}: {count} segments ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print(" ✅ TRANSCRIPTION WITH SPEAKERS COMPLETE!")
    print("="*70)
    
    return matched_segments, speaker_segments


# ==========================================
# QUICK FUNCTION: Get speaker at specific time
# ==========================================

def who_spoke_at(timeline, time_in_seconds):
    """
    Quick function to find who spoke at a specific time
    """
    speaker_intervals = get_speaker_intervals(timeline)
    
    for speaker, intervals in speaker_intervals.items():
        for interval in intervals:
            if interval['start'] <= time_in_seconds <= interval['end']:
                return speaker
    
    return None


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    
    # Your files
    AUDIO_FILE = "audio_file.m4a"
    JSON_FILE = "NS26B_Math-Science Danish Sir_20260220_155725_timeline.json"
    
    print("\n" + "="*70)
    print(" 🎯 SPEECH-TO-TEXT WITH SPEAKER IDENTIFICATION")
    print("="*70)
    
    print("\n   This will:")
    print("   1. Transcribe the audio using Whisper")
    print("   2. Match each transcription segment with the speaker from Zoom")
    print("   3. Show who said what, and when")
    print("")
    
    # Check files
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        exit()
    
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON file not found: {JSON_FILE}")
        exit()
    
    # Run transcription with speaker identification
    matched_segments, speaker_segments = transcribe_with_speakers(
        audio_file=AUDIO_FILE,
        json_file=JSON_FILE,
        model_size="medium"  # Change to "base" for faster, "medium" for better accuracy
    )
    
    # Example: Check who spoke at a specific time
    if matched_segments:
        print("\n" + "="*70)
        print(" 📌 EXAMPLE: Who spoke at specific times?")
        print("="*70)
        
        sample_times = [120, 300, 500, 1000, 1500]
        for t in sample_times:
            if t < len(matched_segments) * 2:  # Rough check
                speaker = who_spoke_at_json(timeline, t) if 'timeline' in locals() else None
                print(f"   At {t}s: {speaker if speaker else 'Unknown'}")