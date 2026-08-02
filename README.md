# Smart Attendance System Using CNN & Face Recognition
An automated AI-powered Smart Attendance System built with OpenCV, Face Recognition, and
Pandas. This project processes group photographs, detects student faces, matches them with
stored embeddings, and logs attendance into Google Drive CSVs.
## Key Features
- **Face DNA Recognition:** Uses deep metric learning (dlib/face_recognition) to identify
multiple students from group photos.
- **Anti-Duplicate Attendance Logging:** Prevents double logging for the same student on
the same date.
- **Time-Window Validation:** Automatically flags records as Present or Late based on
configurable class timing.
- **Visual Analytics:** Displays bounding boxes around detected faces and generates daily
attendance status summary charts.
## Setup & Installation
1. Clone this repository:
```bash
git clone https://github.com/sheetal-tushir/smart-attendance-system.git
```
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook using Google Colab or Jupyter Notebook: `smart_attendance_system.ipynb`
## Workflow
1. **Enrollment:** Upload a clear image of a student to generate and save face encodings
(`.pkl`).
2. **Attendance Processing:** Upload a group photo of the classroom.
3. **Output:** Get marked images with green/red bounding boxes and an auto-updated
`attendance_log.csv` file.
