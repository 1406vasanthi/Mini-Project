import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import face_recognition
import pickle
import pandas as pd
import time

class FaceRecognitionSystem:
    def __init__(self):
        self.dataset_path = "dataset"
        self.model_path = "trained_model.pkl"
        self.attendance_path = "attendance"
        
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
    
    def create_dataset(self, person_name):
        """Create dataset of face images for a person"""
        try:
            if not self.check_camera():
                raise Exception("Camera not available!")
            
            person_dir = os.path.join(self.dataset_path, person_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            count = 0
            total_images = 30
            
            print(f"\nCapturing {total_images} images for {person_name}...")
            print("Please look at the camera and move your face slightly to capture different angles.")
            
            while count < total_images:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame from camera")
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    cv2.putText(frame, f"Progress: {count}/{total_images}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if len(faces) == 1:  # Only save if one face is detected
                        count += 1
                        face_img = frame[y:y+h, x:x+w]
                        file_name = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                        cv2.imwrite(file_name, face_img)
                        time.sleep(0.2)
                
                cv2.imshow('Creating Dataset', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            cap.release()
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"Error in create_dataset: {str(e)}")
            return False

    def train_model(self):
        """Train the face recognition model"""
        try:
            known_faces = []
            known_names = []
            
            for person_name in os.listdir(self.dataset_path):
                person_dir = os.path.join(self.dataset_path, person_name)
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
                        known_names.append(person_name)
            
            model_data = {
                "faces": known_faces,
                "names": known_names
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            return False

    def get_attendance_records(self, date):
        """Get all attendance records for a specific date"""
        date_dir = os.path.join(self.attendance_path, date)
        all_records = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Subject'])
        
        if os.path.exists(date_dir):
            for file in os.listdir(date_dir):
                if file.endswith('_attendance.csv'):
                    file_path = os.path.join(date_dir, file)
                    df = pd.read_csv(file_path)
                    all_records = pd.concat([all_records, df], ignore_index=True)
        
        return all_records

    def mark_attendance(self, subject_name):
        """Mark attendance using face recognition - one attendance per subject per person"""
        try:
            if not self.check_camera():
                raise Exception("Camera not available!")
            
            if not os.path.exists(self.model_path):
                raise Exception("Model not trained! Please train the model first.")
            
            # Load the trained model
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            known_faces = model_data["faces"]
            known_names = model_data["names"]
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            # Create date directory
            date = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(self.attendance_path, date)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            
            # Get all attendance records for today
            attendance_file = os.path.join(date_dir, f"{subject_name}_attendance.csv")
            all_records = self.get_attendance_records(date)
            
            # Get people who already have attendance for this subject today
            subject_attendance = all_records[all_records['Subject'] == subject_name]['Name'].unique()
            
            first_match_found = False
            start_time = datetime.now()
            time_limit = timedelta(seconds=10)
            
            print(f"\nStarting 10-second attendance window for {subject_name}...")
            
            while (datetime.now() - start_time) < time_limit and not first_match_found:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame from camera")
                
                # Find all faces in current frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                # Process only the first detected face in each frame
                if face_encodings:
                    face_encoding = face_encodings[0]
                    face_location = face_locations[0]
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        
                        # Check if person has already been marked for this subject
                        if name not in subject_attendance:
                            time_now = datetime.now().strftime("%H:%M:%S")
                            
                            # Create DataFrame for new entry
                            df = pd.DataFrame([[name, date, time_now, subject_name]], 
                                           columns=['Name', 'Date', 'Time', 'Subject'])
                            
                            # Save to CSV
                            if os.path.exists(attendance_file):
                                df.to_csv(attendance_file, mode='a', header=False, index=False)
                            else:
                                df.to_csv(attendance_file, index=False)
                            
                            print(f"✓ Attendance marked for {name} in {subject_name}")
                            first_match_found = True
                        else:
                            print(f"! {name} already marked for {subject_name} today")
                            first_match_found = True
                    
                    # Draw rectangle and display info
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Display name if matched
                    if True in matches:
                        cv2.putText(frame, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                # Display remaining time and subject
                remaining_time = 10 - int((datetime.now() - start_time).total_seconds())
                cv2.putText(frame, f"Time remaining: {remaining_time}s", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Subject: {subject_name}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Marking Attendance', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if not first_match_found:
                print("\nNo valid face match found during the session.")
            
            return True
            
        except Exception as e:
            print(f"Error in mark_attendance: {str(e)}")
            return False

def display_menu():
    print("\n=== Face Recognition Attendance System ===")
    print("1. Create Dataset")
    print("2. Train Model")
    print("3. Mark Attendance")
    print("4. Exit")
    print("=======================================")

def main():
    system = FaceRecognitionSystem()
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            person_name = input("Enter the person's name for dataset creation: ").strip()
            if person_name:
                if system.create_dataset(person_name):
                    print("Dataset creation completed successfully.")
                else:
                    print("Failed to create dataset. Please check camera connection and try again.")
            else:
                print("Invalid name. Please try again.")

        elif choice == '2':
            print("Training model...")
            if system.train_model():
                print("Model training completed successfully.")
            else:
                print("Failed to train model. Please check if dataset exists.")

        elif choice == '3':
            subject_name = input("Enter the subject name for attendance: ").strip()
            if subject_name:
                if system.mark_attendance(subject_name):
                    print("Attendance marking session ended successfully.")
                else:
                    print("Failed to mark attendance. Please check camera and model.")
            else:
                print("Invalid subject name. Please try again.")

        elif choice == '4':
            print("\nThank you for using the Face Recognition Attendance System!")
            break

        else:
            print("Invalid choice. Please enter a number between 1-4.")

if __name__ == "__main__":
    main()