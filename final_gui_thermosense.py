import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import time

# Function to set the full-screen background image
def set_fullscreen_background(window, canvas, image_path):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    image = Image.open(image_path)
    image = image.resize((screen_width, screen_height), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

# Function to predict the class of the selected image
def predict_class(loaded_model, file_path, result_label, timer_label, start_time):
    test_image = image.load_img(file_path, target_size=(180, 180))
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    class_names = ['overripe', 'partiallyripe', 'ripe', 'unripe']
    output_class = class_names[np.argmax(result)]
    result_label.config(text=f"The predicted class for the selected image is: {output_class}")

    elapsed_time = time.time() - start_time
    timer_label.config(text=f"Time taken: {elapsed_time:.2f} seconds")

# Function to handle image selection and stopwatch timer
def select_image_file(result_label, timer_label):
    result_label.config(text="")  # Clear previous output label text

    start_time = time.time()
    timer_running = True

    def update_timer():
        nonlocal start_time, timer_running
        if timer_running:
            elapsed_time = time.time() - start_time
            timer_label.config(text=f"Time elapsed: {elapsed_time:.2f} seconds")
            timer_label.after(100, update_timer)  # Update every 100 milliseconds

    update_timer()

    initial_directory = r"C:\Users\Parth\Desktop\project\80_20_split\validation"

    file_path = filedialog.askopenfilename(
        initialdir=initial_directory,
        title="Select Image File",
        filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*"))
    )

    if file_path:
        with open(r'C:\Users\Parth\Desktop\project\pickle_model_2', 'rb') as f:
            loaded_model = pickle.load(f)
        
        timer_running = False  # Stop the timer
        predict_class(loaded_model, file_path, result_label, timer_label, start_time)

    else:
        result_label.config(text="No file selected.")

# Create the Tkinter window
window = tk.Tk()
window.attributes('-fullscreen', True)

canvas = tk.Canvas(
    window,
    bg="#FFFFFF",
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.pack(fill=tk.BOTH, expand=True)

set_fullscreen_background(window, canvas, r"C:\Users\Parth\Desktop\project\build\assets\frame0\image_1.png")

result_label = tk.Label(window, text="", font=("Helvetica", 22, "bold"), bg="white")
result_label.place(x=700, y=700)

timer_label = tk.Label(window, text="", font=("Helvetica", 12), bg="white")
timer_label.place(x=1300, y=50)

select_button = tk.Button(
    window,
    text="Select Image",
    command=lambda: select_image_file(result_label, timer_label),
    font=("Helvetica", 12),
    bg="#4CAF50",
    fg="white"
)
select_button.place(x=780, y=250)

window.mainloop()
