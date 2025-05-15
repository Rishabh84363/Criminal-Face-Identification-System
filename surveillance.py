import cv2
import numpy as np
import sqlite3
import face_recognition as fr
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import os
import math
import winsound


class App:
    def __init__(self, video_source=0):
        self.appname = "CFIS- Criminal Face Identification System"
        self.window = Tk()
        self.window.title(self.appname)
        self.window.geometry('1350x720')
        self.window.state("zoomed")
        self.window["bg"] = '#382273'
        self.video_source = video_source
        self.vid = myvideocapture(self.video_source)
        self.label = Label(self.window, text=self.appname, font=("bold", 20), bg='blue', fg='white').pack(side=TOP, fill=BOTH)
        self.canvas = Canvas(self.window, height=700, width=700, bg='#382273')
        self.canvas.pack(side=LEFT, fill=BOTH)
        self.detectedPeople = []
        self.images = self.load_images_from_folder("images")
        self.known_face_names = []
        self.encodings = []

        for img in self.images:
            image_path = os.path.join("images", img)
            try:
                image = fr.load_image_file(image_path)
                encoding = fr.face_encodings(image)
                if encoding:
                    self.encodings.append(encoding[0])
                    # Extract ID from file name (e.g., "user.1.jpg")
                    self.known_face_names.append((os.path.splitext(img)[0]).split('.')[1])
                else:
                    print(f"No face detected in {img}")
            except Exception as e:
                print(f"Error loading {img}: {e}")

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

        print("Loaded IDs:", self.known_face_names)
        self.faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # TreeView setup
        self.tree = ttk.Treeview(self.window, column=("column1", "column2", "column3", "column4", "column5"), show='headings')
        self.tree.heading("#1", text="Cr-ID")
        self.tree.column("#1", minwidth=0, width=70, stretch=NO)
        self.tree.heading("#2", text="NAME")
        self.tree.column("#2", minwidth=0, width=200, stretch=NO)
        self.tree.heading("#3", text="CRIME")
        self.tree.column("#3", minwidth=0, width=150, stretch=NO)
        self.tree.heading("#4", text="Nationality")
        self.tree.column("#4", minwidth=0, width=100, stretch=NO)
        self.tree.heading("#5", text="MATCHING %")
        self.tree.column("#5", minwidth=0, width=120, stretch=NO)

        ttk.Style().configure("Treeview.Heading", font=('Calibri', 13, 'bold'), foreground="red", relief="flat")
        self.tree.place(x=710, y=50)

        self.update()
        self.window.mainloop()

    def load_images_from_folder(self, folder):
        return [filename for filename in os.listdir(folder)]

    def getProfile(self, id):
        conn = sqlite3.connect("criminal.db")
        cmd = "SELECT ID, name, crime, nationality FROM people WHERE ID = ?"
        cursor = conn.execute(cmd, (id,))
        profile = cursor.fetchone()
        conn.close()
        return profile

    def showPercentageMatch(self, face_distance, face_match_threshold=0.6):
        if face_distance > face_match_threshold:
            range = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

    def update(self):
        isTrue, frame = self.vid.getframe()
        if isTrue:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

            # Face detection
            self.face_locations = fr.face_locations(frame)
            self.face_encodings = fr.face_encodings(frame, self.face_locations)
            self.face_names = []

            for face_encoding in self.face_encodings:
                matches = fr.compare_faces(self.encodings, face_encoding)
                face_distance = fr.face_distance(self.encodings, face_encoding)
                name = "Unknown"

                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    face_percent = self.showPercentageMatch(face_distance[best_match_index])

                    if face_percent > 0.50:  # Adjust threshold as needed
                        self.face_names.append(name)
                        self.detectedPeople.append(name)
                        profile = self.getProfile(int(name))
                        if profile:
                            self.tree.insert("", "end", values=(profile[0], profile[1], profile[2], profile[3], f"{face_percent * 100:.2f}%"))
                        self.play_alert_sound()

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

            if len(self.detectedPeople) > 0:
                print("Detected people:", self.detectedPeople)
            self.detectedPeople = []

        self.window.after(10, self.update)

    def play_alert_sound(self):
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)


class myvideocapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def getframe(self):
        ret, frame = self.vid.read()
        if not ret:
            return (False, None)
        return (True, frame)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == '__main__':
    App()
