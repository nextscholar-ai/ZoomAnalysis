import json
from collections import defaultdict



def time_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)




with open(
    "NS26B_Math-Science Danish Sir_20260220_155725_timeline.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

timeline = data["timeline"]


user_is_teacher = {}


teacher_student_interactions = defaultdict(int)


last_speaker = None
last_speaker_end_time = 0
previous_timestamp = None
previous_users = set()

for entry in timeline:
    current_timestamp = time_to_seconds(entry["ts"])
    
    # Get current users
    current_users = set()
    for user in entry.get("users", []):
        username = user.get("username", "").strip()
        zoom_userid = user.get("zoom_userid", "").strip()
        
        if not username:
            continue
        
        current_users.add(username)
        
        # Store whether this user is a teacher (has zoom_userid)
        if username not in user_is_teacher:
            user_is_teacher[username] = bool(zoom_userid)
    
    # Detect speaker changes
    if len(current_users) > 0:
        # Find the main speaker (prioritize teacher if multiple)
        current_speaker = None
        for user in current_users:
            if user_is_teacher.get(user, False):
                current_speaker = "TEACHER"
                break
        if not current_speaker:
            current_speaker = list(current_users)[0]
        
        # Check if speaker changed
        if last_speaker is not None and last_speaker != current_speaker:
            # Interaction between teacher and student
            if (last_speaker == "TEACHER" and current_speaker != "TEACHER") or \
               (last_speaker != "TEACHER" and current_speaker == "TEACHER"):
                # Count interaction for the student
                student_name = current_speaker if current_speaker != "TEACHER" else last_speaker
                teacher_student_interactions[student_name] += 1
        
        last_speaker = current_speaker
    
    previous_users = current_users
    previous_timestamp = current_timestamp


# OUTPUT


print("\n========== TEACHER-STUDENT INTERACTIONS ==========\n")

# Sort by number of interactions (highest first)
sorted_interactions = sorted(teacher_student_interactions.items(), key=lambda x: x[1], reverse=True)

for student, count in sorted_interactions:
    print(f"{student} → {count} time(s)")

print("\n===================================================")