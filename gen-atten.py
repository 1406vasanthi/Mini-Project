import streamlit as st
import cv2
import numpy as np
from datetime import datetime, timedelta
import face_recognition
import pickle
import pandas as pd
import time
import os
from PIL import Image
import base64

# Add function to set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Backend Classes remain the same
class Timetable:
    # [Previous Timetable class code remains exactly the same]
    def __init__(self):
        # Define periods with their timings
        self.periods = {
            1: ("09:30", "10:30"),
            2: ("10:30", "11:20"),
            3: ("11:20", "12:10"),
            4: ("12:10", "13:00"),
            5: ("13:00", "14:00"),
            6: ("14:00", "14:50"),
            7: ("14:50", "15:40"),
            8: ("15:40", "16:30")
        }
        
        # Sample timetable data structure - ensure all periods (1-8) are included
        self.schedule = {
            'Monday': {1: 'PCS-III(Aptitude)', 2: 'PCS-III(Aptitude)', 3: 'Artificial Intelligence(AI)', 
                      4: 'Data Mining(DM)', 5: 'LUNCH', 6: 'MCCP-I', 7: 'MCCP-I', 8: 'MCCP-I'},
            'Tuesday': {1: 'Operating Systems(OS)', 2: 'DATA MINING LAB', 3: 'DATA MINING LAB', 
                       4: 'DATA MINING LAB', 5: 'LUNCH', 6: 'Web Technologies(WT)', 7: 'Artificial Intelligence(AI)', 
                       8: 'Data Mining(DM)'},
            'Wednesday': {1: 'Data Mining(DM)', 2: 'Web Technologies(WT)', 3: 'Web Technologies(WT)', 
                         4: 'Library', 5: 'LUNCH', 6: 'Artificial Intelligence(AI)', 7: 'Operating Systems(OS)', 
                         8: 'Sports'},
            'Thursday': {1: 'Artificial Intelligence(AI)', 2: 'Operating Systems(OS)', 3: 'Data Mining(DM)', 
                        4: 'Operating Systems(OS)', 5: 'LUNCH', 6: 'Web Technologies Lab', 7: 'Web Technologies Lab', 
                        8: 'Web Technologies Lab'},
            'Friday': {1: 'Operating Systems(OS)', 2: 'Data Mining(DM)', 3: 'Web Technologies(WT)', 
                      4: 'Artificial Intelligence(AI)', 5: 'LUNCH', 6: 'MCCP-I', 7: 'MCCP-I', 8: 'MCCP-I'},
            'Saturday': {1: 'Web Technologies(WT)', 2: 'Data Mining(DM)', 3: 'PCS-III(Verbal)', 
                        4: 'PCS-III(Verbal)', 5: 'LUNCH', 6: 'Artificial Intelligence(AI)', 7: 'Operating Systems(OS)', 
                        8: 'Web Technologies(WT)'},
            'Sunday': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
        }
        
        # Faculty information
        self.faculty = {
            'PCS-III(Aptitude)': 'J N V SOMAYAJULU',
            'Artificial Intelligence(AI)': 'S KUMAR REDDY MALLIDI',
            'Data Mining(DM)': 'G.PRASANTHI',
            'MCCP-I': 'Dr. V VENKATESWARA RAO, KADALI RAMYA',
            'Operating Systems(OS)': 'A LEELAVATHI',
            'PCS-III(Verbal)': 'AMARLAPUDI KIRANMAYEE',
            'DATA MINING LAB': 'G.PRASANTHI,G SRI RAM GANESH',
            'Web Technologies(WT)': 'L ATRI DATTA RAVITEZ',
            'Web Technologies Lab': 'L ATRI DATTA RAVITEZ,YENTRAPATI SABITHA YALI',
            'Library': 'Library Staff',
            'Sports': 'Sports Department',
            'LUNCH': 'Break Time'
        }

    def get_current_period(self):
        """Get current period based on time"""
        current_time = datetime.now()
        current_day = current_time.strftime('%A')
        current_time_str = current_time.strftime('%H:%M')
        
        if current_day == 'Sunday':
            return None, None, None
            
        for period, (start, end) in self.periods.items():
            if start <= current_time_str <= end:
                subject_code = self.schedule[current_day][period]
                faculty_info = self.faculty.get(subject_code, "No faculty assigned")
                return period, subject_code, faculty_info
                
        return None, None, None

class StudentManagementSystem:
    # [Previous StudentManagementSystem class code remains exactly the same]
    def __init__(self):
        self.dataset_path = "dataset"
        self.model_path = "trained_model.pkl"
        self.attendance_path = "attendance"
        self.timetable = Timetable()
        
        # Create necessary directories
        for path in [self.dataset_path, self.attendance_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def check_camera(self):
        """Test camera availability"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return False
        cap.release()
        return True

    def train_model(self):
        """Train the face recognition model"""
        try:
            known_faces = []
            known_names = []
            
            for person_folder in os.listdir(self.dataset_path):
                person_dir = os.path.join(self.dataset_path, person_folder)
                if os.path.isdir(person_dir):
                    person_encodings = []
                    for image_name in os.listdir(person_dir):
                        image_path = os.path.join(person_dir, image_name)
                        image = face_recognition.load_image_file(image_path)
                        
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            person_encodings.append(face_encodings[0])
                    
                    if person_encodings:
                        average_encoding = np.mean(person_encodings, axis=0)
                        known_faces.append(average_encoding)
                        known_names.append(person_folder)
            
            model_data = {
                "faces": known_faces,
                "names": known_names
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            st.error(f"Error in train_model: {str(e)}")
            return False

# Modified main function with styling
def main():
    st.set_page_config(page_title="Student Management System", layout="wide")
    
    # Add background image
    # Make sure to create an 'assets' folder and place your background image there
    add_bg_from_local('assets/hm2.jpg')
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main-title {
            color: #FFFFFF;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px 4px #000000;
            padding: 20px;
        }
        .tagline {
            color: #FFFFFF;
            font-size: 24px;
            text-align: center;
            font-style: italic;
            text-shadow: 1px 1px 2px #000000;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: rgba(255, 255, 255, 0.1);
            border: 2px solid white;
        }
        .stSelectbox, .stTextInput {
            background-color: rgba(255, 255, 255, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize system in session state
    if 'system' not in st.session_state:
        st.session_state.system = StudentManagementSystem()
    
    # Sidebar navigation with styling
    with st.sidebar:
        st.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: rgba(255, 255, 255, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        st.title("Navigation")
        page = st.radio(
            "",
            ["Home", "Register Student", "Mark Attendance", "View Schedule", "Attendance Reports"]
        )
    
    # Main content
    st.markdown('<h1 class="main-title">Next-Gen Attendance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Elevate Your Attendance Experience</p>', unsafe_allow_html=True)
    
    # Page routing
    if page == "Home":
        display_home()
    elif page == "Register Student":
        register_student_page()
    elif page == "Mark Attendance":
        mark_attendance_page()
    elif page == "View Schedule":
        view_schedule_page()
    elif page == "Attendance Reports":
        attendance_reports_page()

# [Rest of the function implementations remain exactly the same]
def display_home():
    st.header("Welcome to Next-Gen Attendance")
    
    # Display current class information
    st.subheader("Current Class Information")
    period, subject_code, faculty_info = st.session_state.system.timetable.get_current_period()
    
    if period and subject_code:
        start_time, end_time = st.session_state.system.timetable.periods[period]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Period {period}")
        with col2:
            st.info(f"Time: {start_time} - {end_time}")
        with col3:
            st.info(f"Subject: {subject_code}")
        st.info(f"Faculty: {faculty_info}")
    else:
        st.warning("No class scheduled at this time.")

def register_student_page():
    st.header("Register New Student")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_name = st.text_input("Student Name")
        roll_number = st.text_input("Roll Number")
        
        if st.button("Start Registration"):
            if not student_name or not roll_number:
                st.error("Please enter both name and roll number!")
                return
                
            st.session_state.registration_active = True
            st.session_state.capture_count = 0
            st.session_state.student_name = student_name
            st.session_state.roll_number = roll_number
    
    with col2:
        if st.session_state.get('registration_active', False):
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            
            try:
                person_dir = os.path.join(st.session_state.system.dataset_path, 
                                        f"{student_name}_{roll_number}")
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)
                
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while st.session_state.capture_count < 30:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from camera")
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        if len(faces) == 1:
                            st.session_state.capture_count += 1
                            face_img = frame[y:y+h, x:x+w]
                            file_name = os.path.join(person_dir, f"img_{st.session_state.capture_count}.jpg")
                            cv2.imwrite(file_name, face_img)
                            time.sleep(0.2)
                    
                    # Update progress
                    progress = st.session_state.capture_count / 30
                    progress_bar.progress(progress)
                    status_text.text(f"Capturing: {st.session_state.capture_count}/30 images")
                    
                    # Display the frame
                    stframe.image(convert_cv2_to_pil(frame), channels="RGB")
                    
                cap.release()
                
                if st.session_state.capture_count >= 30:
                    st.success("Registration completed successfully!")
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            if st.session_state.system.train_model():
                                st.success("Model trained successfully!")
                            else:
                                st.error("Error in model training")
                
            except Exception as e:
                st.error(f"Error during registration: {str(e)}")
            finally:
                st.session_state.registration_active = False

def mark_attendance_page():
    st.header("Mark Attendance")
    
    period, subject_code, faculty_info = st.session_state.system.timetable.get_current_period()
    
    if not subject_code:
        st.warning("No ongoing class at this time.")
        return
    
    st.info(f"Current Class: {subject_code}")
    st.info(f"Faculty: {faculty_info}")
    
    if st.button("Start Attendance"):
        if not os.path.exists(st.session_state.system.model_path):
            st.error("Model not trained! Please train the model first.")
            return
            
        stframe = st.empty()
        status_text = st.empty()
        stop_button = st.empty()
        
        try:
            with open(st.session_state.system.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            known_faces = model_data["faces"]
            known_names = model_data["names"]
            
            cap = cv2.VideoCapture(0)
            marked_students = set()
            
            # Create attendance directory and file
            date = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(st.session_state.system.attendance_path, date)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            
            attendance_file = os.path.join(date_dir, f"{subject_code}_attendance.csv")
            
            stop_attendance = stop_button.button("Stop Attendance")
            
            while not stop_attendance:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from camera")
                    break
                
                # Find faces in current frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        
                        # Mark attendance if not already marked
                        if name not in marked_students:
                            time_now = datetime.now().strftime("%H:%M:%S")
                            
                            df = pd.DataFrame([[name, date, time_now, subject_code]], 
                                           columns=['Name', 'Date', 'Time', 'Subject'])
                            
                            if os.path.exists(attendance_file):
                                df.to_csv(attendance_file, mode='a', header=False, index=False)
                            else:
                                df.to_csv(attendance_file, index=False)
                            
                            marked_students.add(name)
                            status_text.success(f"✓ Attendance marked for {name}")
                        
                        # Draw rectangle and name
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name.split('_')[0], (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                # Display the frame
                stframe.image(convert_cv2_to_pil(frame), channels="RGB")
                stop_attendance = stop_button.button("Stop Attendance")
                
            cap.release()
            st.success("Attendance marking completed!")
            
        except Exception as e:
            st.error(f"Error in attendance marking: {str(e)}")

def view_schedule_page():
    st.header("Class Schedule")
    
    tab1, tab2 = st.tabs(["Current Schedule", "Full Timetable"])
    
    with tab1:
        st.subheader("Current Schedule")
        period, subject_code, faculty_info = st.session_state.system.timetable.get_current_period()
        
        if period and subject_code:
            start_time, end_time = st.session_state.system.timetable.periods[period]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"Period {period}")
            with col2:
                st.info(f"Time: {start_time} - {end_time}")
            with col3:
                st.info(f"Subject: {subject_code}")
            st.info(f"Faculty: {faculty_info}")
        else:
            st.warning("No class scheduled at this time.")
    
    with tab2:
        st.subheader("Full Weekly Schedule")
        # Convert timetable to DataFrame for better display
        schedule_data = []
        for period, (start, end) in st.session_state.system.timetable.periods.items():
            row = {'Period': f"Period {period}", 'Time': f"{start}-{end}"}
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                row[day] = st.session_state.system.timetable.schedule[day][period]
            schedule_data.append(row)
        
        df = pd.DataFrame(schedule_data)
        st.dataframe(df, use_container_width=True)

def attendance_reports_page():
    st.header("Attendance Reports")
    
    # Date selection
    selected_date = st.date_input(
        "Select Date",
        value=datetime.now()
    ).strftime("%Y-%m-%d")
    
    date_dir = os.path.join(st.session_state.system.attendance_path, selected_date)
    
    if os.path.exists(date_dir):
        all_records = pd.DataFrame()
        for file in os.listdir(date_dir):
            if file.endswith('_attendance.csv'):
                file_path = os.path.join(date_dir, file)
                df = pd.read_csv(file_path)
                all_records = pd.concat([all_records, df], ignore_index=True)
        
        if not all_records.empty:
            # Display summary
            st.subheader("Attendance Summary")
            for subject in all_records['Subject'].unique():
                with st.expander(f"Subject: {subject}"):
                    subject_records = all_records[all_records['Subject'] == subject]
                    st.info(f"Total Students Present: {len(subject_records)}")
                    st.dataframe(subject_records)
        else:
            st.warning("No attendance records found for selected date.")
    else:
        st.warning("No attendance records found for selected date.")

if __name__ == "__main__":
    main()