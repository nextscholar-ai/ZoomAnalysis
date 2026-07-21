import os
import json
import re
import numpy as np
from collections import defaultdict
import ollama
from datetime import datetime

MODEL_NAME = "richardyoung/llama-3.2-3b-instruct-abliterated:Q4_K_M"
JSON_FILE = "audio_file_with_speakers.json"
AUDIO_FILE = "audio_file.wav"
OUTPUT_FILE = "communication_analysis_report.txt"


def extract_pitch_features(audio_file):
    """Extract pitch and frequency features from audio file"""
    try:
        import librosa
        import numpy as np
        
        print("   🎵 Loading audio for pitch analysis...")
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        times = librosa.times_like(f0, sr=sr)
        pitch_data = []
        
        window_size = int(sr * 1.0)
        
        for i in range(0, len(y), window_size):
            start_time = i / sr
            end_time = min((i + window_size) / sr, len(y) / sr)
            
            window_indices = np.where((times >= start_time) & (times < end_time))
            window_pitch = f0[window_indices]
            valid_pitch = window_pitch[~np.isnan(window_pitch)]
            
            if len(valid_pitch) > 0:
                pitch_mean = np.mean(valid_pitch)
                pitch_std = np.std(valid_pitch)
                pitch_min = np.min(valid_pitch)
                pitch_max = np.max(valid_pitch)
                pitch_range = pitch_max - pitch_min
                energy = np.mean(y[i:i+window_size]**2)
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=y[i:i+window_size], sr=sr
                ).mean()
                
                if pitch_mean > 200 and pitch_range > 50:
                    sentiment = "Positive"
                    pitch_score = 1.0
                elif pitch_mean < 120 and pitch_range < 30:
                    sentiment = "Negative"
                    pitch_score = -1.0
                elif pitch_mean >= 150 and pitch_range > 40:
                    sentiment = "Positive"
                    pitch_score = 0.7
                elif pitch_mean < 140:
                    sentiment = "Negative"
                    pitch_score = -0.5
                else:
                    sentiment = "Neutral"
                    pitch_score = 0
                
                pitch_data.append({
                    'start': start_time,
                    'end': end_time,
                    'pitch_mean': float(pitch_mean),
                    'pitch_std': float(pitch_std),
                    'pitch_range': float(pitch_range),
                    'energy': float(energy),
                    'spectral_centroid': float(spectral_centroid),
                    'sentiment': sentiment,
                    'pitch_score': pitch_score
                })
        
        print(f"   ✓ Extracted pitch features for {len(pitch_data)} segments")
        return pitch_data
        
    except ImportError:
        print("   ⚠️ librosa not installed. Install with: pip install librosa")
        return None
    except Exception as e:
        print(f"   ⚠️ Error in pitch analysis: {e}")
        return None

def get_pitch_sentiment(timestamp, pitch_data, window=2.0):
    """Get sentiment from pitch data at a specific timestamp"""
    if not pitch_data:
        return None, 0
    
    best_match = None
    best_distance = float('inf')
    
    for pitch in pitch_data:
        mid_time = (pitch['start'] + pitch['end']) / 2
        distance = abs(mid_time - timestamp)
        
        if distance < window and distance < best_distance:
            best_distance = distance
            best_match = pitch
    
    if best_match:
        return best_match['sentiment'], best_match['pitch_score']
    else:
        return None, 0



def is_disturbance_text(text):
    """Check if text contains disturbance patterns"""
    disturbance_patterns = [
        'Hindi and English mixed speech conversation',
        'Hindi and English mixed',
        'speech conversation',
        '[Music]',
        '[Applause]',
        '[Silence]',
        '...',
        '. . .',
        'speech recognition',
        'transcription',
    ]
    
    text_lower = text.lower().strip()
    
    for pattern in disturbance_patterns:
        if pattern.lower() in text_lower:
            return True
    
    if len(text) < 3:
        return True
    
    return False

def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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

def extract_teacher_from_filename(json_file):
    """Extract teacher name from filename"""
    base_name = os.path.basename(json_file)
    base_name = os.path.splitext(base_name)[0]
    
    patterns = [
        r'(?:[A-Z0-9]+_)?(?:[A-Za-z-]+ )?([A-Za-z\s]+) (?:Sir|Madam|Ma\'am|Teacher)',
        r'([A-Za-z\s]+) (?:Sir|Madam|Ma\'am|Teacher)_\d+',
        r'Teacher[_\s]+([A-Za-z_]+)',
        r'^([A-Za-z\s]+)(?:_\d+|\.\w+)?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name, re.IGNORECASE)
        if match:
            teacher_name = match.group(1).strip()
            teacher_name = re.sub(r'[_\s]+', ' ', teacher_name).strip()
            return teacher_name
    
    return None

def identify_teacher(segments, json_file):
    """Identify the teacher from filename or data"""
    teacher_name = extract_teacher_from_filename(json_file)
    
    speakers = set()
    for seg in segments:
        if 'speaker' in seg:
            speakers.add(seg['speaker'])
    
    if teacher_name:
        for speaker in speakers:
            if teacher_name.lower() in speaker.lower() or speaker.lower() in teacher_name.lower():
                return speaker
    
    common_teachers = ['Danish', 'Teacher', 'Sir', 'Madam']
    for teacher in common_teachers:
        for speaker in speakers:
            if teacher.lower() in speaker.lower():
                return speaker
    
    speaker_counts = defaultdict(int)
    for seg in segments:
        speaker = seg.get('speaker', '')
        if speaker:
            speaker_counts[speaker] += 1
    
    if speaker_counts:
        return max(speaker_counts, key=speaker_counts.get)
    
    return "Teacher"



def classify_communication(segments, teacher_name, pitch_data):
    """Classify each communication segment using both text and audio"""
    
    results = {
        'teacher': [],
        'students': defaultdict(list),
        'all_segments': []
    }
    
    clean_segments = []
    for seg in segments:
        text = seg.get('text', '').strip()
        if not text or is_disturbance_text(text):
            continue
        
        clean_text_segment = clean_text(text)
        if len(clean_text_segment) < 3:
            continue
        
        clean_segments.append({
            'speaker': seg.get('speaker', 'Unknown'),
            'text': clean_text_segment,
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'is_teacher': seg.get('speaker', '') == teacher_name
        })
    
    if not clean_segments:
        print("   ⚠️ No valid segments found after cleaning")
        return results
    
    print(f"   ✓ Filtered to {len(clean_segments)} valid segments")
    
    batch_size = 30
    for i in range(0, len(clean_segments), batch_size):
        batch = clean_segments[i:i+batch_size]
        
        for item in batch:
            mid_time = (item['start'] + item['end']) / 2
            audio_sentiment, audio_score = get_pitch_sentiment(mid_time, pitch_data)
            item['audio_sentiment'] = audio_sentiment
            item['audio_score'] = audio_score
        
        prompt = """
You are a communication analyst. Classify each message as Positive, Negative, or Neutral based on the tone and content.

Rules:
- Positive: Encouraging, supportive, appreciative, constructive, enthusiastic
- Negative: Critical, harsh, dismissive, frustrated, discouraging
- Neutral: Informational, factual, procedural, neither positive nor negative

Format your response exactly as:
1. [Classification] - [1-2 sentence reason]
2. [Classification] - [1-2 sentence reason]
... and so on for all messages.

Messages to classify:
"""
        
        for j, item in enumerate(batch, 1):
            prompt += f"\n{j}. {item['speaker']}: {item['text']}"
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a communication analyst. Provide clear, brief classifications with reasons.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            text_classifications = parse_classifications(response['message']['content'], batch)
            
            for item, text_class in zip(batch, text_classifications):
                final_sentiment = text_class['sentiment']
                final_reason = text_class['reason']
                
                if item['audio_sentiment'] and item['audio_sentiment'] != final_sentiment:
                    if item['audio_score'] > 0.5 or item['audio_score'] < -0.5:
                        final_sentiment = item['audio_sentiment']
                        final_reason = f"{text_class['reason']} (Audio: {item['audio_sentiment']} tone)"
                
                result = {
                    'speaker': item['speaker'],
                    'text': item['text'],
                    'start': item['start'],
                    'end': item['end'],
                    'sentiment': final_sentiment,
                    'reason': final_reason,
                    'is_teacher': item['is_teacher'],
                    'audio_sentiment': item['audio_sentiment'],
                    'audio_score': item['audio_score']
                }
                
                results['all_segments'].append(result)
                
                if item['is_teacher']:
                    results['teacher'].append(result)
                else:
                    results['students'][item['speaker']].append(result)
        
        except Exception as e:
            print(f"⚠️ Error classifying batch: {e}")
            for item in batch:
                final_sentiment = item['audio_sentiment'] if item['audio_sentiment'] else 'Neutral'
                result = {
                    'speaker': item['speaker'],
                    'text': item['text'],
                    'start': item['start'],
                    'end': item['end'],
                    'sentiment': final_sentiment,
                    'reason': 'Fallback classification due to API error',
                    'is_teacher': item['is_teacher'],
                    'audio_sentiment': item['audio_sentiment'],
                    'audio_score': item['audio_score']
                }
                results['all_segments'].append(result)
                if item['is_teacher']:
                    results['teacher'].append(result)
                else:
                    results['students'][item['speaker']].append(result)
    
    return results

def parse_classifications(response, batch):
    """Parse the classification response from LLM"""
    classifications = []
    
    lines = response.strip().split('\n')
    line_index = 0
    
    for i, item in enumerate(batch):
        found = False
        while line_index < len(lines):
            line = lines[line_index].strip()
            if not line:
                line_index += 1
                continue
            
            match = re.match(r'^\s*(\d+)\.\s*([A-Za-z]+)\s*[-:]\s*(.+)', line, re.IGNORECASE)
            if match:
                sentiment = match.group(2).strip().capitalize()
                reason = match.group(3).strip()
                
                if sentiment not in ['Positive', 'Negative', 'Neutral']:
                    for s in ['Positive', 'Negative', 'Neutral']:
                        if s.lower() in line.lower():
                            sentiment = s
                            break
                    else:
                        sentiment = 'Neutral'
                
                classifications.append({
                    'sentiment': sentiment,
                    'reason': reason
                })
                found = True
                line_index += 1
                break
            else:
                sentiment_found = None
                for s in ['Positive', 'Negative', 'Neutral']:
                    if s.lower() in line.lower():
                        sentiment_found = s
                        break
                
                if sentiment_found:
                    parts = re.split(f'(?i){sentiment_found}[-:]*', line, maxsplit=1)
                    reason = parts[1].strip() if len(parts) > 1 else line.strip()
                    classifications.append({
                        'sentiment': sentiment_found,
                        'reason': reason[:100]
                    })
                    found = True
                    line_index += 1
                    break
            
            line_index += 1
        
        if not found:
            classifications.append({
                'sentiment': 'Neutral',
                'reason': 'Classification not found in response'
            })
    
    while len(classifications) < len(batch):
        classifications.append({
            'sentiment': 'Neutral',
            'reason': 'Fallback classification'
        })
    
    return classifications[:len(batch)]


def generate_systematic_report(classifications, analysis, teacher_name, json_file):
    """Generate a systematic, well-formatted report"""
    
    report_lines = []
    
  
    report_lines.append("=" * 100)
    report_lines.append(" " * 30 + "COMMUNICATION ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f" Source File: {json_file}")
    report_lines.append(f" Teacher: {teacher_name}")
    report_lines.append("=" * 100)
    report_lines.append("")
    
  
    report_lines.append("=" * 100)
    report_lines.append(" SECTION 1: EXECUTIVE SUMMARY")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    total = analysis['overall']['total']
    if total > 0:
        pos_pct = (analysis['overall']['positive'] / total) * 100
        neg_pct = (analysis['overall']['negative'] / total) * 100
        neu_pct = (analysis['overall']['neutral'] / total) * 100
        
        report_lines.append("┌─────────────┬──────────┬────────────┬──────────────┐")
        report_lines.append("│ METRIC      │ POSITIVE │ NEGATIVE   │ NEUTRAL      │")
        report_lines.append("├─────────────┼──────────┼────────────┼──────────────┤")
        report_lines.append(f"│ Overall     │ {analysis['overall']['positive']:>6}   │ {analysis['overall']['negative']:>7}    │ {analysis['overall']['neutral']:>10}    │")
        report_lines.append(f"│ Percentage  │ {pos_pct:>6.1f}%   │ {neg_pct:>7.1f}%    │ {neu_pct:>10.1f}%    │")
        report_lines.append("└─────────────┴──────────┴────────────┴──────────────┘")
        report_lines.append("")
        
        # Overall Sentiment Assessment
        if pos_pct > 40:
            overall_sentiment = "POSITIVE - Communication was generally constructive and encouraging"
        elif neg_pct > 30:
            overall_sentiment = "NEGATIVE - Significant critical or discouraging communication detected"
        else:
            overall_sentiment = "NEUTRAL - Balanced communication with mixed sentiments"
        
        report_lines.append(f" Overall Sentiment Assessment: {overall_sentiment}")
        report_lines.append("")
        
        # Audio Analysis Summary
        if analysis['audio_analysis_summary']['total_audio_segments'] > 0:
            report_lines.append(" Audio Analysis Summary:")
            report_lines.append(f"   • Audio segments analyzed: {analysis['audio_analysis_summary']['total_audio_segments']}")
            report_lines.append(f"   • Matches with text sentiment: {analysis['audio_analysis_summary']['matches_with_text']}")
            report_lines.append(f"   • Disagreements: {analysis['audio_analysis_summary']['disagreements']}")
            report_lines.append("")
    
    report_lines.append("=" * 100)
    report_lines.append(" SECTION 2: TEACHER COMMUNICATION ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    teacher_total = analysis['teacher']['total']
    if teacher_total > 0:
        pos_count = len(analysis['teacher']['positive'])
        neg_count = len(analysis['teacher']['negative'])
        neu_count = len(analysis['teacher']['neutral'])
        
        report_lines.append("┌─────────────┬──────────┬────────────┬──────────────┐")
        report_lines.append("│ TEACHER     │ POSITIVE │ NEGATIVE   │ NEUTRAL      │")
        report_lines.append("├─────────────┼──────────┼────────────┼──────────────┤")
        report_lines.append(f"│ Messages    │ {pos_count:>6}   │ {neg_count:>7}    │ {neu_count:>10}    │")
        report_lines.append(f"│ Percentage  │ {(pos_count/teacher_total*100):>6.1f}%   │ {(neg_count/teacher_total*100):>7.1f}%    │ {(neu_count/teacher_total*100):>10.1f}%    │")
        report_lines.append("└─────────────┴──────────┴────────────┴──────────────┘")
        report_lines.append("")
        
        # Positive Examples
        if analysis['teacher']['positive']:
            report_lines.append(" ✅ POSITIVE COMMUNICATION EXAMPLES:")
            report_lines.append(" ─" * 60)
            for idx, msg in enumerate(analysis['teacher']['positive'][:5], 1):
                audio_info = f" (Audio: {msg.get('audio_sentiment', 'N/A')})" if msg.get('audio_sentiment') else ""
                report_lines.append(f"   {idx}. [{msg['start']:.0f}s - {msg['end']:.0f}s]{audio_info}")
                report_lines.append(f"      Reason: {msg['reason']}")
                report_lines.append(f"      Text: \"{msg['text'][:100]}\"")
                report_lines.append("")
        
        # Negative Examples
        if analysis['teacher']['negative']:
            report_lines.append(" ❌ NEGATIVE COMMUNICATION EXAMPLES:")
            report_lines.append(" ─" * 60)
            for idx, msg in enumerate(analysis['teacher']['negative'][:5], 1):
                audio_info = f" (Audio: {msg.get('audio_sentiment', 'N/A')})" if msg.get('audio_sentiment') else ""
                report_lines.append(f"   {idx}. [{msg['start']:.0f}s - {msg['end']:.0f}s]{audio_info}")
                report_lines.append(f"      Reason: {msg['reason']}")
                report_lines.append(f"      Text: \"{msg['text'][:100]}\"")
                report_lines.append("")
    else:
        report_lines.append(" No teacher messages found.")
        report_lines.append("")
    
    report_lines.append("=" * 100)
    report_lines.append(" SECTION 3: STUDENT COMMUNICATION ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    if not analysis['students']:
        report_lines.append(" No student messages found.")
        report_lines.append("")
    
    # Student Summary Table
    report_lines.append("┌─────────────────────┬──────────┬──────────┬──────────┬──────────────┐")
    report_lines.append("│ STUDENT             │ POSITIVE │ NEGATIVE │ NEUTRAL  │ TOTAL        │")
    report_lines.append("├─────────────────────┼──────────┼──────────┼──────────┼──────────────┤")
    
    for student, data in sorted(analysis['students'].items(), key=lambda x: x[1]['total'], reverse=True):
        pos = len(data['positive'])
        neg = len(data['negative'])
        neu = len(data['neutral'])
        total_student = data['total']
        
        # Truncate long names
        display_name = student[:20] + "..." if len(student) > 20 else student
        report_lines.append(f"│ {display_name:<19} │ {pos:>6}   │ {neg:>7} │ {neu:>7} │ {total_student:>8}   │")
    
    report_lines.append("└─────────────────────┴──────────┴──────────┴──────────┴──────────────┘")
    report_lines.append("")
    
    # Individual Student Detailed Analysis
    for student, data in sorted(analysis['students'].items(), key=lambda x: x[1]['total'], reverse=True):
        if data['total'] == 0:
            continue
        
        report_lines.append("─" * 100)
        report_lines.append(f" STUDENT: {student}")
        report_lines.append("─" * 100)
        report_lines.append("")
        
        pos = len(data['positive'])
        neg = len(data['negative'])
        neu = len(data['neutral'])
        total_student = data['total']
        
        report_lines.append(f" 📊 Communication Statistics:")
        report_lines.append(f"    Total Messages: {total_student}")
        report_lines.append(f"    Positive: {pos} ({(pos/total_student*100):.1f}%)")
        report_lines.append(f"    Negative: {neg} ({(neg/total_student*100):.1f}%)")
        report_lines.append(f"    Neutral: {neu} ({(neu/total_student*100):.1f}%)")
        
        # Overall Sentiment
        if pos > neg and pos > neu:
            overall = "🟢 POSITIVE"
        elif neg > pos and neg > neu:
            overall = "🔴 NEGATIVE"
        else:
            overall = "🟡 NEUTRAL"
        report_lines.append(f"    Overall Sentiment: {overall}")
        report_lines.append("")
        
        # Positive Examples
        if data['positive']:
            report_lines.append(" ✅ Positive Examples:")
            for idx, msg in enumerate(data['positive'][:3], 1):
                audio_info = f" (Audio: {msg.get('audio_sentiment', 'N/A')})" if msg.get('audio_sentiment') else ""
                report_lines.append(f"    {idx}. [{msg['start']:.0f}s]{audio_info}")
                report_lines.append(f"       \"{msg['text'][:80]}\"")
                report_lines.append(f"       → {msg['reason']}")
                report_lines.append("")
        
        # Negative Examples
        if data['negative']:
            report_lines.append(" ❌ Negative Examples:")
            for idx, msg in enumerate(data['negative'][:3], 1):
                audio_info = f" (Audio: {msg.get('audio_sentiment', 'N/A')})" if msg.get('audio_sentiment') else ""
                report_lines.append(f"    {idx}. [{msg['start']:.0f}s]{audio_info}")
                report_lines.append(f"       \"{msg['text'][:80]}\"")
                report_lines.append(f"       → {msg['reason']}")
                report_lines.append("")
        
        # Detailed Analysis from LLM
        report_lines.append(" 📝 Detailed Communication Analysis:")
        detailed_lines = generate_student_detailed_analysis(student, data, classifications)
        for line in detailed_lines:
            report_lines.append(f"    {line}")
        report_lines.append("")
    
    report_lines.append("=" * 100)
    report_lines.append(" SECTION 4: TIMELINE SENTIMENT ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    # Create sentiment timeline
    all_segments = sorted(classifications['all_segments'], key=lambda x: x['start'])
    
    report_lines.append(" Sentiment Timeline (First 20 significant exchanges):")
    report_lines.append(" ─" * 80)
    report_lines.append("")
    
    count = 0
    for seg in all_segments:
        if count >= 20:
            break
        
        sentiment_emoji = "🟢" if seg['sentiment'] == 'Positive' else "🔴" if seg['sentiment'] == 'Negative' else "🟡"
        speaker_type = "👨‍🏫" if seg['is_teacher'] else "👤"
        audio_info = f" (Audio: {seg.get('audio_sentiment', 'N/A')})" if seg.get('audio_sentiment') else ""
        
        report_lines.append(f" [{seg['start']:.0f}s] {sentiment_emoji} {speaker_type} {seg['speaker']}{audio_info}")
        report_lines.append(f"    {seg['text'][:80]}")
        report_lines.append(f"    → {seg['reason']}")
        report_lines.append("")
        count += 1
    
    report_lines.append("=" * 100)
    report_lines.append(" SECTION 5: RECOMMENDATIONS")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    recommendations = generate_recommendations(analysis, teacher_name)
    for rec in recommendations:
        report_lines.append(f" • {rec}")
        report_lines.append("")
    
    report_lines.append("=" * 100)
    report_lines.append(" " * 40 + "END OF REPORT")
    report_lines.append("=" * 100)
    
    return report_lines

def generate_student_detailed_analysis(student_name, student_data, classifications):
    """Generate detailed analysis for a specific student using LLM"""
    
    lines = []
    
    all_messages = student_data['positive'] + student_data['negative'] + student_data['neutral']
    all_messages.sort(key=lambda x: x['start'])
    
    if not all_messages:
        lines.append("No messages to analyze.")
        return lines
    
    messages_text = ""
    for msg in all_messages[:15]:
        audio_info = f" [Audio: {msg.get('audio_sentiment', 'N/A')}]" if msg.get('audio_sentiment') else ""
        messages_text += f"\n[{msg['start']:.0f}s]{audio_info} {msg['sentiment']}: {msg['text']}"
    
    prompt = f"""
Analyze this student's communication pattern in detail:

Student: {student_name}
Total Messages: {len(all_messages)}
Sentiment Breakdown:
- Positive: {len(student_data['positive'])}
- Negative: {len(student_data['negative'])}
- Neutral: {len(student_data['neutral'])}

Sample Messages:
{messages_text}

Provide a detailed analysis covering:
1. Communication Style Summary (2-3 sentences)
2. Key Strengths (what they do well)
3. Areas for Improvement (what could be better)
4. Specific Recommendations (actionable advice)
5. Overall Communication Assessment (1-2 sentences)

Format your response with these clear headings.
"""
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are an educational communication expert. Provide clear, detailed, and constructive analysis.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        analysis_text = response['message']['content']
        
        for line in analysis_text.split('\n'):
            if line.strip():
                if line.strip().startswith('1.') or line.strip().startswith('2.') or \
                   line.strip().startswith('3.') or line.strip().startswith('4.') or \
                   line.strip().startswith('5.'):
                    lines.append(f"{line}")
                elif line.strip().startswith('Key') or line.strip().startswith('Areas') or \
                     line.strip().startswith('Specific') or line.strip().startswith('Overall'):
                    lines.append(f"{line}")
                else:
                    lines.append(f"   {line}")
    
    except Exception as e:
        lines.append(f"Error generating detailed analysis: {e}")
    
    return lines

def generate_recommendations(analysis, teacher_name):
    """Generate recommendations based on sentiment analysis"""
    
    recommendations = []
    
    # Teacher recommendations
    teacher_total = analysis['teacher']['total']
    teacher_neg = len(analysis['teacher']['negative'])
    teacher_pos = len(analysis['teacher']['positive'])
    
    if teacher_total > 0:
        if teacher_neg / teacher_total > 0.3:
            recommendations.append(
                f"The teacher ({teacher_name}) has a high proportion of negative communication "
                f"({teacher_neg}/{teacher_total}). Consider using more encouraging and constructive language."
            )
        elif teacher_pos / teacher_total > 0.5:
            recommendations.append(
                f"The teacher ({teacher_name}) demonstrates excellent positive communication "
                f"({teacher_pos}/{teacher_total}). Continue this supportive approach."
            )
        else:
            recommendations.append(
                f"The teacher's communication is balanced. Focus on maintaining clarity "
                f"and increasing positive reinforcement."
            )
    
    # Student recommendations
    for student, data in analysis['students'].items():
        total = data['total']
        neg = len(data['negative'])
        pos = len(data['positive'])
        
        if total > 0:
            if neg / total > 0.4:
                recommendations.append(
                    f"Student '{student}' shows significant negative communication "
                    f"({neg}/{total}). Consider providing additional support and encouragement."
                )
            elif pos / total < 0.2 and total > 10:
                recommendations.append(
                    f"Student '{student}' has low positive engagement. Encourage more active participation."
                )
    
    # General recommendations
    if analysis['overall']['negative'] > analysis['overall']['positive']:
        recommendations.append(
            "Overall negative sentiment detected. Consider reviewing teaching methods "
            "and addressing student concerns."
        )
    elif analysis['overall']['neutral'] > 0.6 * analysis['overall']['total']:
        recommendations.append(
            "High proportion of neutral communication. Consider adding more interactive "
            "elements to increase engagement."
        )
    
    if not recommendations:
        recommendations.append(
            "Communication appears well-balanced. Continue current teaching strategies "
            "and monitor for any changes."
        )
    
    return recommendations[:5]  # Limit to top 5 recommendations

def analyze_sentiment_patterns(classifications):
    """Analyze sentiment patterns and generate summary statistics"""
    
    analysis = {
        'teacher': {
            'positive': [],
            'negative': [],
            'neutral': [],
            'total': 0
        },
        'students': {},
        'overall': {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0
        },
        'audio_analysis_summary': {
            'total_audio_segments': 0,
            'matches_with_text': 0,
            'disagreements': 0
        }
    }
    
    for msg in classifications['all_segments']:
        sentiment = msg['sentiment'].lower()
        analysis['overall'][sentiment] += 1
        analysis['overall']['total'] += 1
        
        if msg.get('audio_sentiment'):
            analysis['audio_analysis_summary']['total_audio_segments'] += 1
            if msg['audio_sentiment'].lower() == sentiment:
                analysis['audio_analysis_summary']['matches_with_text'] += 1
            else:
                analysis['audio_analysis_summary']['disagreements'] += 1
    
    for msg in classifications['teacher']:
        sentiment = msg['sentiment'].lower()
        analysis['teacher'][sentiment].append(msg)
        analysis['teacher']['total'] += 1
    
    for student, messages in classifications['students'].items():
        analysis['students'][student] = {
            'positive': [],
            'negative': [],
            'neutral': [],
            'total': 0
        }
        
        for msg in messages:
            sentiment = msg['sentiment'].lower()
            analysis['students'][student][sentiment].append(msg)
            analysis['students'][student]['total'] += 1
    
    return analysis



def main():
    print("\n" + "=" * 100)
    print(" " * 30 + "COMMUNICATION SENTIMENT ANALYZER")
    print("=" * 100)
    
    try:
        ollama.list()
        print("✅ Ollama is running")
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        return
    
    print(f"\n📂 Loading transcript: {JSON_FILE}")
    segments = load_transcript(JSON_FILE)
    
    if not segments:
        print("❌ No data found")
        return
    
    print(f"   ✓ Loaded {len(segments)} segments")
    
    pitch_data = None
    if os.path.exists(AUDIO_FILE):
        print(f"\n🎵 Analyzing audio: {AUDIO_FILE}")
        pitch_data = extract_pitch_features(AUDIO_FILE)
    else:
        print(f"\n⚠️ Audio file not found: {AUDIO_FILE}")
        print("   Continuing with text-only analysis")
    
    print(f"\n🔍 Identifying teacher...")
    teacher_name = identify_teacher(segments, JSON_FILE)
    print(f"   ✓ Teacher identified: {teacher_name}")
    
    print(f"\n🤖 Classifying communication sentiment...")
    classifications = classify_communication(segments, teacher_name, pitch_data)
    
    if not classifications or not classifications['all_segments']:
        print("❌ No classifications generated")
        return
    
    print(f"   ✓ Classified {len(classifications['all_segments'])} messages")
    print(f"   - Teacher: {len(classifications['teacher'])} messages")
    print(f"   - Students: {sum(len(v) for v in classifications['students'].values())} messages")
    
    print("\n📊 Analyzing sentiment patterns...")
    analysis = analyze_sentiment_patterns(classifications)
    
    print("\n📝 Generating systematic report...")
    report_lines = generate_systematic_report(classifications, analysis, teacher_name, JSON_FILE)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + "\n")
    
    print(f"\n✅ Report saved to: {OUTPUT_FILE}")
    print("=" * 100)
    print(" " * 35 + "ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()