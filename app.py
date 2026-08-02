import streamlit as st
import face_recognition
import cv2
import pickle
import os
import pandas as pd
from datetime import datetime, time
import numpy as np

st.set_page_config(page_title="Smart Attendance System", layout="wide")
st.title("📸 Smart AI Attendance System")

# Ensure local storage directory exists
os.makedirs("model", exist_ok=True)
db_path = 'model/student_database.pkl'
csv_path = 'model/attendance_log.csv'

# Initialize database if missing
if not os.path.exists(db_path):
    with open(db_path, 'wb') as f:
        pickle.dump({'encodings': [], 'names': []}, f)

# Navigation
menu = st.sidebar.selectbox("Menu", ["Mark Attendance", "Register New Student", "View Logs"])

if menu == "Register New Student":
    st.subheader("👤 Student Enrollment")
    student_name = st.text_input("Enter Student Name & Roll No")
    uploaded_file = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])
    
    if st.button("Register Student"):
        if student_name and uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) > 0:
                with open(db_path, 'rb') as f:
                    database = pickle.load(f)
                database['encodings'].append(encodings[0])
                database['names'].append(student_name)
                
                with open(db_path, 'wb') as f:
                    pickle.dump(database, f)
                st.success(f"✅ Successfully registered {student_name}!")
            else:
                st.error("❌ No face detected in the image. Try again.")
        else:
            st.warning("Please provide name and photo.")

elif menu == "Mark Attendance":
    st.subheader("🎯 Mark Class Attendance")
    group_file = st.file_uploader("Upload Classroom Group Photo", type=["jpg", "jpeg", "png"])
    
    if st.button("Process Attendance"):
        if group_file:
            with open(db_path, 'rb') as f:
                database = pickle.load(f)
            known_face_encodings = database['encodings']
            known_names = database['names']
            
            if not known_names:
                st.error("No registered students found! Register students first.")
            else:
                file_bytes = np.asarray(bytearray(group_file.read()), dtype=np.uint8)
                group_img = cv2.imdecode(file_bytes, 1)
                rgb_group_img = cv2.cvtColor(group_img, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_group_img)
                face_encodings = face_recognition.face_encodings(rgb_group_img, face_locations)
                
                recognized_students = set()
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"
                    color = (255, 0, 0)
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        color = (0, 255, 0)
                        recognized_students.add(name)
                    
                    cv2.rectangle(rgb_group_img, (left, top), (right, bottom), color, 2)
                    cv2.putText(rgb_group_img, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                status = "Present" if time(9, 30) <= now.time() <= time(13, 30) else "Late"
                
                if os.path.exists(csv_path):
                    df_existing = pd.read_csv(csv_path)
                else:
                    df_existing = pd.DataFrame(columns=["Date", "Time", "Name / Roll No", "Status"])
                
                new_records = []
                for student in recognized_students:
                    already_marked = not df_existing[(df_existing['Date'] == date_str) & (df_existing['Name / Roll No'] == student)].empty
                    if not already_marked:
                        new_records.append({"Date": date_str, "Time": time_str, "Name / Roll No": student, "Status": status})
                
                if new_records:
                    df_final = pd.concat([df_existing, pd.DataFrame(new_records)], ignore_index=True)
                    df_final.to_csv(csv_path, index=False)
                    st.success("✅ Attendance Logged Successfully!")
                else:
                    st.info("Attendance already logged for recognized students today.")
                
                st.image(rgb_group_img, caption="Processed Image", use_container_width=True)

elif menu == "View Logs":
    st.subheader("📋 Attendance Log")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)
    else:
        st.info("No attendance logged yet.")
