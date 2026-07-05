import json
from collections import Counter



with open(
    "NS26B_Math-Science Danish Sir_20260220_155725_timeline.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

timeline = data["timeline"]


user_count = Counter()

for entry in timeline:
    for user in entry.get("users", []):

        name = user.get("username", "").strip()

        if name:
            user_count[name] += 1

teacher_name = user_count.most_common(1)[0][0]



student_names = set()

for entry in timeline:
    for user in entry.get("users", []):

        name = user.get("username", "").strip()

        if not name:
            continue

        # Skip teacher
        if name == teacher_name:
            continue

        # Add unique students
        student_names.add(name)

# ==========================================
# OUTPUT
# ==========================================

print("\n========== CLASS ANALYSIS ==========")

print("\nTeacher Name:", teacher_name)

print("\nTotal Active Students:", len(student_names))

print("\nActive Student Names:")

for student in student_names:
    print("-", student)

print("\n====================================")