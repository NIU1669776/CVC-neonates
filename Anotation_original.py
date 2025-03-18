import tkinter as tk
from tkinter import filedialog, messagebox
import os
import random
import csv
from PIL import Image, ImageTk

# Global variables
image_paths = []
clicks = []
current_image = None
csv_file = "test_set_2.csv"

# Load all valid image paths
def load_image_paths(root_folder):
    paths = []
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path) and folder.isdigit() and 29 <= int(folder) <= 55:
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for file in os.listdir(subfolder_path):
                        if file.endswith(".jpeg") and (".VIS.jpeg" in file or "_VIS.jpeg" in file):
                            paths.append(os.path.join(subfolder_path, file))
    return paths

# Load a new random image
def load_random_image():
    global current_image, clicks
    if not image_paths:
        messagebox.showerror("Error", "No valid images found!")
        return
    
    clicks = []
    img_path = random.choice(image_paths)
    current_image = img_path
    img = Image.open(img_path)
    img = img.resize((816, 612), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    canvas.image = img_tk
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    update_status()

# Handle mouse clicks
def on_click(event):
    global clicks
    if len(clicks) < 17:
        clicks.append((event.x, event.y))
    update_status()

# Undo last click
def undo_last_click():
    if clicks:
        clicks.pop()
    update_status()

# Skip a point
def skip_point():
    if len(clicks) < 17:
        clicks.append("")  # Empty string for skipped point
    update_status()

# Update status label
def update_status():
    labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear','left_wrist',
            'right_wrist', 'left_elbow', 'right_elbow','left_knee', 'right_knee',
            'left_ankle', 'right_ankle','left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip']
    status_text = " | ".join(f"{labels[i]}: {clicks[i]}" if i < len(clicks) else f"{labels[i]}: ?" for i in range(17))
    status_label.config(text=status_text)
    
    if len(clicks) == 17:
        save_to_csv()
        load_random_image()

# Save to CSV
def save_to_csv():
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=';')
        row = [current_image] + clicks
        writer.writerow(row)

# Initialize the GUI
root = tk.Tk()
root.title("Image Annotation Tool")

canvas = tk.Canvas(root, width=816, height=612)
canvas.pack()
canvas.bind("<Button-1>", on_click)

status_label = tk.Label(root, text="Click on the image to annotate points", pady=10)
status_label.pack()

button_frame = tk.Frame(root)
button_frame.pack()

undo_button = tk.Button(button_frame, text="Undo Last Click", command=undo_last_click)
skip_button = tk.Button(button_frame, text="Skip", command=skip_point)
undo_button.pack(side=tk.LEFT, padx=10)
skip_button.pack(side=tk.RIGHT, padx=10)

# Start the app
image_paths = load_image_paths("images")
load_random_image()
root.mainloop()