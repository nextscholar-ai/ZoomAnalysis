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

user_speaking_start = {}

user_total_duration = defaultdict(float)

user_is_teacher = {}


previous_timestamp = None
previous_users = set()

for entry in timeline:
    current_timestamp = time_to_seconds(entry["ts"])
    
    # Get current users (who have microphone ON)
    current_users = set()
    for user in entry.get("users", []):
        username = user.get("username", "").strip()
        zoom_userid = user.get("zoom_userid", "").strip()
        
        if not username: 
            continue
        
        current_users.add(username)
        
        
        if username not in user_is_teacher:
            user_is_teacher[username] = bool(zoom_userid)  
    
    if previous_timestamp is not None:
        stopped_users = previous_users - current_users
        
        for user in stopped_users:
            if user in user_speaking_start:
                duration = current_timestamp - user_speaking_start[user]
                user_total_duration[user] += duration
                del user_speaking_start[user]
    
    started_users = current_users - previous_users
    
    for user in started_users:
        user_speaking_start[user] = current_timestamp
    
    previous_users = current_users
    previous_timestamp = current_timestamp


for user, start_time in user_speaking_start.items():
    if previous_timestamp is not None:
        duration = previous_timestamp - start_time
        user_total_duration[user] += duration



def format_time(sec):
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}m {seconds}s"



teacher_duration = 0
student_durations = []

for user, duration in user_total_duration.items():
    if user_is_teacher.get(user, False):  
        teacher_duration = duration
    else:  
        student_durations.append((user, duration))


student_durations.sort(key=lambda x: x[1], reverse=True)

print(f"\nTeacher Duration:")
print(f"{format_time(teacher_duration)}")

print("\n========== STUDENT TOTAL DURATIONS ==========")

for student, duration in student_durations:
    print(f"{student} → {format_time(duration)}")
