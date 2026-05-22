Here’s a professional `README.md` for your GitHub project based on your internship work and Python files.

# 🎓 AI-Based Classroom Conversation Analysis System

An AI-powered classroom analytics system developed during my internship to analyze online class recordings and interaction data.
The project focuses on identifying active participants, calculating class duration, tracking teacher–student conversation time, and analyzing classroom engagement using Python.

---

# 🚀 Features

* ⏱️ Calculate total class duration
* 🎤 Detect active teacher–student conversations
* 👨‍🏫 Identify teacher and active students
* 🗣️ Analyze speaking duration of participants
* 📊 Generate classroom engagement insights
* 🔍 Process timeline JSON data from recorded classes
* 🎧 Analyze audio interaction using speech activity detection

---

# 📁 Project Files

## 1️⃣ `Duration_of_class.py`

Calculates:

* Total duration of the online class
* Start and end timestamps
* Overall session length

### Functionality

* Reads timeline JSON data
* Converts timestamps into seconds
* Computes total class duration

---

## 2️⃣ `duration_of_conversesion.py`

Analyzes:

* Teacher speaking duration
* Student conversation duration
* Total interaction time between teacher and students

### Functionality

* Detects active speaking sessions
* Tracks mic ON/OFF duration
* Calculates total conversation time for each participant

---

## 3️⃣ `indetification_of_studnet.py`

Identifies:

* Teacher name
* Total active students
* Unique student participants

### Functionality

* Extracts usernames from timeline data
* Removes duplicate entries
* Separates teacher and student activity

---

# 🛠️ Technologies Used

* Python 3
* JSON Processing
* Audio Processing
* Pydub
* FFmpeg
* Data Analysis Techniques

---

# 📂 Input Data

The system works with:

* `.json` timeline files
* `.m4a` classroom audio recordings

---

# ⚙️ Installation

## Install Required Libraries

```bash
pip install pydub
pip install ffmpeg-python
```

---

# 🔧 Install FFmpeg

Download FFmpeg:

[https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)

Add:

```text
ffmpeg/bin
```

to your system PATH.

---

# ▶️ Run the Project

```bash
python Duration_of_class.py
```

```bash
python duration_of_conversesion.py
```

```bash
python indetification_of_studnet.py
```

---

# 📊 Example Output

```text
Teacher Name: Danish Hayat

Total Active Students: 4

Student Names:
- KHAN KULSOOM
- Khan Zainab
- Sufiyan
- 𝑧𝑜ℎ𝑑𝑎𝑘ℎ𝑎𝑡𝑜𝑜𝑛
```

---

# 🎯 Project Objectives

This project aims to:

* Improve classroom analytics
* Measure student engagement
* Analyze online teaching effectiveness
* Build AI-driven educational insights

---

# 🔮 Future Improvements

* 🤖 AI-based teacher rating system
* 🧠 NLP-based question-answer analysis
* 📈 Dashboard visualization
* 📄 Automatic Excel report generation
* 🎤 Advanced speaker diarization
* 😊 Emotion and sentiment analysis

---

# 👨‍💻 Developed By

Ashish Lale
Artificial Intelligence & Data Science Student
Savitribai Phule Pune University (SPPU)

---

# 📌 Internship Project

Developed as part of an AI-based classroom analytics internship project focused on educational interaction analysis and engagement monitoring.
