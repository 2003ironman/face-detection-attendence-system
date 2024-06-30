# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:28:27 2024

@author: Aditya Singh
"""

import tkinter as tk
from tkinter import messagebox as mess
import cv2
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from PIL import Image, ImageTk
import shutil

# Path to attendance system folder
attendance_system_path = "C:\\Users\\Aditya Singh\\Desktop\\attendence system"

# Ensure directory and file creation
def ensure_directory_and_file_exists():
    directory = attendance_system_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create or update attendance.csv
    csv_path = os.path.join(directory, 'attendance.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Name', 'ID', 'Branch', 'Email', 'Date', 'Time'])

    # Create or update attendance.xlsx using pandas
    excel_path = os.path.join(directory, 'attendance.xlsx')
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=['Name', 'ID', 'Branch', 'Email', 'Date', 'Time'])
        df.to_excel(excel_path, index=False)

    # Ensure TrainingImage and RegisteredUsers directories exist
    train_img_path = os.path.join(directory, 'TrainingImage')
    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)

    registered_users_path = os.path.join(directory, 'RegisteredUsers')
    if not os.path.exists(registered_users_path):
        os.makedirs(registered_users_path)

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Function to decode base64 to image
def decode_base64_to_image(encoded_string):
    decoded_image = base64.b64decode(encoded_string)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return img

# Function to clear entry fields
def clear_entries():
    txt_id.delete(0, 'end')
    txt_name.delete(0, 'end')
    txt_branch.delete(0, 'end')
    txt_email.delete(0, 'end')
    message_label.config(text="")
    clear_camera_frame()

# Function to clear camera frame
def clear_camera_frame():
    camera_label.config(image='')

# Function to update attendance for a new day
def update_attendance_for_new_day():
    csv_path = os.path.join(attendance_system_path, 'attendance.csv')
    try:
        df = pd.read_csv(csv_path)
        today_date = datetime.now().strftime('%Y-%m-%d')
        
        for index, row in df.iterrows():
            last_registered_date = row['Date']
            if last_registered_date != today_date:
                df.at[index, 'Date'] = today_date
        
        df.to_csv(csv_path, index=False)
    
    except Exception as e:
        print(f"Error updating attendance for new day: {str(e)}")

# Function to save details and image to file
def save_details_to_file(name, Id, branch, email, image_path, time_registered):
    img = cv2.imread(image_path)
    cv2.putText(img, f"Name: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"ID: {Id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Branch: {branch}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Email: {email}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Time Registered: {time_registered}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    encoded_image = encode_image_to_base64(image_path)
    
    # Save image details to CSV
    csv_path = os.path.join(attendance_system_path, 'attendance.csv')
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, Id, branch, email, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S')])
    
    # Save image details to Excel using pandas
    excel_path = os.path.join(attendance_system_path, 'attendance.xlsx')
    df = pd.DataFrame({
        'Name': [name],
        'ID': [Id],
        'Branch': [branch],
        'Email': [email],
        'Date': [datetime.now().strftime('%Y-%m-%d')],
        'Time': [datetime.now().strftime('%H:%M:%S')]
    })
    
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        df.to_excel(excel_path, index=False)
    
    # Save image with details
    save_path = os.path.join(attendance_system_path, "RegisteredUsers")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, f"{name}_{Id}.jpg"), img)

# Function to train images
def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces, Ids = get_images_and_labels(os.path.join(attendance_system_path, "TrainingImage"))
    recognizer.train(faces, np.array(Ids))
    recognizer.save(os.path.join(attendance_system_path, 'TrainingImageLabel/Trainner.yml'))
    print("Training Complete")

# Function to get images and labels for training
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    
    for image_path in image_paths:
        if os.path.split(image_path)[-1].split(".")[-1] == 'jpg':
            pil_image = Image.open(image_path).convert('L')  # convert it to grayscale
            image_np = np.array(pil_image, 'uint8')
            Id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(image_np)
            Ids.append(Id)
    
    return faces, Ids

# Function to take images and register users
def take_images():
    Id = txt_id.get()
    name = txt_name.get()
    branch = txt_branch.get()
    email = txt_email.get()
    
    if Id == '' or name == '' or branch == '' or email == '':
        message_label.config(text="Please enter ID, Name, Branch, and Email")
    else:
        try:
            Id = int(Id)
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            sample_num = 0
            
            ensure_directory_and_file_exists()
            
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    sample_num += 1
                    image_path = os.path.join(attendance_system_path, "TrainingImage", f"{name}.{Id}.jpg")
                    cv2.imwrite(image_path, gray[y:y + h, x:x + w])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, f'{str(Id)}-{name}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    break  # Take only the first face detected
                
                small_frame = cv2.resize(img, (320, 240))
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(small_frame))
                camera_label.config(image=photo)
                camera_label.image = photo  # Keep a reference
                
                cv2.imshow('Taking Images', img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sample_num > 1:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            time_registered = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_details_to_file(name, Id, branch, email, image_path, time_registered)
            clear_entries()
            train_images()
            message_label.config(text="Images Saved Successfully")
        
        except Exception as e:
            message_label.config(text=f"Error: {str(e)}")

# Function to mark attendance
def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(attendance_system_path, 'TrainingImageLabel/Trainner.yml'))
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    df = pd.read_csv(os.path.join(attendance_system_path, 'attendance.csv'))
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 75:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                name = df.loc[df['ID'] == Id]['Name'].values[0]
                email = df.loc[df['ID'] == Id]['Email'].values[0]
                branch = df.loc[df['ID'] == Id]['Branch'].values[0]
                attendance_details = [name, Id, branch, email, ts]
                df.loc[df['ID'] == Id, ['Name', 'Branch', 'Email', 'Date', 'Time']] = attendance_details
                df.to_csv(os.path.join(attendance_system_path, 'attendance.csv'), index=False)
                
                cv2.putText(img, str(Id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            else:
                Id = 'Unknown'
                cv2.putText(img, str(Id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        
        small_frame = cv2.resize(img, (320, 240))
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(small_frame))
        camera_label.config(image=photo)
        camera_label.image = photo  # Keep a reference
        
        cv2.imshow('Marking Attendance', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

# Function to setup UI
def setup_ui():
    global txt_id, txt_name, txt_branch, txt_email, message_label, camera_label
    
    window = tk.Tk()
    window.title("Attendance System")
    window.geometry('1280x720')
    window.configure(bg='light green')
    
    lbl_title = tk.Label(window, text="Face Recognition Attendance System", bg='light green', fg='black', font=('times', 24, 'bold'))
    lbl_title.place(x=400, y=20)
    
    lbl_id = tk.Label(window, text="Enter ID", bg='light green', fg='black', font=('times', 15, 'bold'))
    lbl_id.place(x=200, y=100)
    txt_id = tk.Entry(window, width=20, bg='white', fg='black', font=('times', 15, 'bold'))
    txt_id.place(x=400, y=100)
    
    lbl_name = tk.Label(window, text="Enter Name", bg='light green', fg='black', font=('times', 15, 'bold'))
    lbl_name.place(x=200, y=150)
    txt_name = tk.Entry(window, width=20, bg='white', fg='black', font=('times', 15, 'bold'))
    txt_name.place(x=400, y=150)
    
    lbl_branch = tk.Label(window, text="Enter Branch", bg='light green', fg='black', font=('times', 15, 'bold'))
    lbl_branch.place(x=200, y=200)
    txt_branch = tk.Entry(window, width=20, bg='white', fg='black', font=('times', 15, 'bold'))
    txt_branch.place(x=400, y=200)
    
    lbl_email = tk.Label(window, text="Enter Email", bg='light green', fg='black', font=('times', 15, 'bold'))
    lbl_email.place(x=200, y=250)
    txt_email = tk.Entry(window, width=20, bg='white', fg='black', font=('times', 15, 'bold'))
    txt_email.place(x=400, y=250)
    
    message_label = tk.Label(window, text="", bg='light green', fg='red', font=('times', 15, 'bold'))
    message_label.place(x=400, y=300)
    
    camera_label = tk.Label(window, bg='light green')
    camera_label.place(x=700, y=100)
    
    btn_take_img = tk.Button(window, text="Take Image", command=take_images, fg='black', bg='white', width=20, height=2, font=('times', 15, 'bold'))
    btn_take_img.place(x=200, y=350)
    
    btn_mark_attendance = tk.Button(window, text="Mark Attendance", command=mark_attendance, fg='black', bg='white', width=20, height=2, font=('times', 15, 'bold'))
    btn_mark_attendance.place(x=450, y=350)
    
    btn_clear = tk.Button(window, text="Clear", command=clear_entries, fg='black', bg='white', width=20, height=2, font=('times', 15, 'bold'))
    btn_clear.place(x=700, y=350)
    
    window.mainloop()

if __name__ == "__main__":
    ensure_directory_and_file_exists()
    setup_ui()
