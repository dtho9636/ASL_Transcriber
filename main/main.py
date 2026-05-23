"""
ASL Sign Language Transcriber  |  TSA Nationals - Software Development
Sandy Utah, Team #1: Daxton Thorne, Claire Kunz, Rayhan Ahmed, Carmine Terry, Adelaide Hopkins

A real-time American Sign Language transcriber that uses MediaPipe's hand
landmarker model to detect and classify hand gestures.

How to run the program from scratch:
    1. Install Python 3.10 or newer.
    2. Create and activate a virtual environment:
         python -m venv .venv
         .venv\\Scripts\\Activate.ps1   # PowerShell
         .venv\\Scripts\\activate.bat   # Command Prompt
    3. Install dependencies:
         pip install opencv-python mediapipe numpy Pillow
    4. Create a folder named "data" next to the parent folder of this script.
       For example, if this file is in "main/", then create "data/" beside "main/".
    5. Download the MediaPipe hand landmarker model file and put it here:
         data/hand_landmarker.task
    6. Run the program:
         python main.py

Controls (in the GUI):
    • "Collect Sample" button  — capture the current gesture with a custom label
    • "Clear Transcript"       — wipe the running transcript text
    • "Quit" / close window    — exit the application
"""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image


# Model implementation ------------------------------------------------------------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SAMPLE_FILE = os.path.join(DATA_DIR, "sign_samples.npz")
MODEL_FILE = os.path.join(DATA_DIR, "hand_landmarker.task")

# Helpers

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def find_model_path():
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE
    try:
        import mediapipe as mp
    except ImportError:
        return None
    root = os.path.dirname(mp.__file__)
    for dirpath, _, filenames in os.walk(root):
        if "hand_landmarker.task" in filenames:
            return os.path.join(dirpath, "hand_landmarker.task")
    return None

# Feature extraction and sample storage

def normalize_landmarks(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z], dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    relative = coords - wrist
    scale = np.max(np.abs(relative))
    if scale > 0:
        relative /= scale
    return relative.flatten()


def load_samples():
    if not os.path.exists(SAMPLE_FILE):
        return None, None
    try:
        data = np.load(SAMPLE_FILE, allow_pickle=True)
        samples = data["samples"]
        labels = data["labels"].tolist()
        if len(samples) != len(labels):
            return None, None
        return samples, labels
    except Exception:
        return None, None


def save_samples(samples, labels):
    ensure_data_dir()
    np.savez_compressed(
        SAMPLE_FILE,
        samples=np.array(samples, dtype=np.float32),
        labels=np.array(labels, dtype=object),
    )


def predict_from_samples(vector, samples, labels, k=3, max_distance=0.6):
    if samples is None or labels is None or len(samples) == 0:
        return None
    distances = np.linalg.norm(samples - vector, axis=1)
    nearest = np.argsort(distances)[:k]
    if len(nearest) == 0 or distances[nearest[0]] > max_distance:
        return None
    nearest_labels = [labels[idx] for idx in nearest]
    counts = {}
    for label in nearest_labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)

# Gesture heuristic labels

def fingers_status(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    status = []
    for tip_id, pip_id in zip(tips, pips):
        status.append(landmarks[tip_id].y < landmarks[pip_id].y)
    return status


def heuristic_label(landmarks):
    status = fingers_status(landmarks)
    thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended = status

    tip = lambda i: np.array([landmarks[i].x, landmarks[i].y])
    wrist     = tip(0)
    thumb_tip = tip(4)
    index_tip = tip(8)
    pinky_tip = tip(20)

    dist_thumb_index = np.linalg.norm(thumb_tip - index_tip)
    dist_thumb_pinky = np.linalg.norm(thumb_tip - pinky_tip)
    dist_wrist_index = np.linalg.norm(wrist - index_tip)

    if index_extended and middle_extended and ring_extended and not pinky_extended and not thumb_extended:
        return "W"
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "V"
    if pinky_extended and not index_extended and not middle_extended and not ring_extended:
        return "I"
    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Thumb"
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "D"
    if thumb_extended and pinky_extended and not index_extended and not middle_extended and not ring_extended:
        return "Y"
    if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "L"
    if index_extended and not middle_extended and not ring_extended and pinky_extended and not thumb_extended:
        return "1"
    if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        if dist_thumb_index < 0.06 and dist_wrist_index < 0.12:
            return "A"
        return "S"
    if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "P"
    if thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
        return "B"
    if index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended:
        return "B"
    if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
        return "B"
    if dist_thumb_index < 0.05 and dist_thumb_pinky < 0.05 and index_extended and middle_extended and ring_extended and pinky_extended:
        return "O"
    return None

# MediaPipe creation

def create_landmarker():
    model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Download hand_landmarker.task and place it in the data/ folder."
        )
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return hand_landmarker.HandLandmarker.create_from_options(options)

# Tkinter ------------------------------------------------------------------------------------------------------------------------------

BG_DARK   = "#1a1a2e"
BG_PANEL  = "#16213e"
BG_CARD   = "#0f3460"
ACCENT    = "#e94560"
ACCENT2   = "#53d8fb"
TEXT_MAIN = "#eaeaea"
TEXT_DIM  = "#8892a4"
GREEN     = "#4ade80"
FONT_MONO = ("Courier", 13)


class ASLTranscriberApp:
    # Main window

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ASL Sign Language Transcriber")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # State
        self.running          = True
        self.cap              = None
        self.landmarker       = None
        self.timestamp_ms     = 0
        self.last_prediction  = None
        self.last_append_time = 0.0
        self.samples          = None
        self.labels           = None
        self.latest_frame     = None
        self.latest_landmarks = None
        self.latest_vector    = None
        self._collect_pending = False

        self._build_ui()
        self._load_data()
        self._init_camera_and_model()

    # UI ------------------------------------------------------------------------------------------------------------------------------

    def _build_ui(self):

        # Header
        header = tk.Frame(self.root, bg=BG_DARK, pady=10)
        header.pack(fill="x")

        tk.Label(
            header,
            text="ASL Sign Language Transcriber",
            font=("Helvetica", 18, "bold"),
            bg=BG_DARK, fg=ACCENT,
        ).pack(side="left", padx=20)

        tk.Label(
            header,
            text="Team #1  •  TSA Nationals",
            font=("Helvetica", 10),
            bg=BG_DARK, fg=TEXT_DIM,
        ).pack(side="right", padx=20)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        # Main
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=15, pady=10)

        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both")

        cam_border = tk.Frame(left, bg=ACCENT, bd=2, relief="flat")
        cam_border.pack()
        self.video_label = tk.Label(cam_border, bg="black", cursor="crosshair")
        self.video_label.pack()

        right = tk.Frame(body, bg=BG_DARK, padx=12)
        right.pack(side="left", fill="both", expand=True)

        # Current sign card
        sign_card = tk.Frame(right, bg=BG_CARD, padx=12, pady=12,
                             relief="flat", bd=0)
        sign_card.pack(fill="x", pady=(0, 10))

        tk.Label(sign_card, text="Detected Sign",
                 font=("Helvetica", 10), bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")

        self.sign_var = tk.StringVar(value="—")
        tk.Label(
            sign_card,
            textvariable=self.sign_var,
            font=("Helvetica", 52, "bold"),
            bg=BG_CARD, fg=GREEN,
            width=6,
        ).pack()

        self.method_var = tk.StringVar(value="")
        tk.Label(
            sign_card,
            textvariable=self.method_var,
            font=("Helvetica", 9),
            bg=BG_CARD, fg=TEXT_DIM,
        ).pack()

        # Status card
        status_card = tk.Frame(right, bg=BG_PANEL, padx=12, pady=8,
                               relief="flat", bd=0)
        status_card.pack(fill="x", pady=(0, 10))

        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(
            status_card,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            bg=BG_PANEL, fg=ACCENT2,
            wraplength=260, justify="left",
        ).pack(anchor="w")

        # Samples counter
        self.sample_count_var = tk.StringVar(value="Samples loaded: 0")
        tk.Label(
            status_card,
            textvariable=self.sample_count_var,
            font=("Helvetica", 9),
            bg=BG_PANEL, fg=TEXT_DIM,
        ).pack(anchor="w", pady=(4, 0))

        # Transcript box
        tk.Label(right, text="Live Transcript",
                 font=("Helvetica", 10, "bold"),
                 bg=BG_DARK, fg=TEXT_MAIN).pack(anchor="w")

        transcript_frame = tk.Frame(right, bg=BG_PANEL, bd=1, relief="sunken")
        transcript_frame.pack(fill="both", expand=True, pady=(2, 10))

        self.transcript_text = tk.Text(
            transcript_frame,
            height=6, width=28,
            font=FONT_MONO,
            bg=BG_PANEL, fg=GREEN,
            insertbackground=GREEN,
            relief="flat", wrap="word",
            state="disabled",
        )
        self.transcript_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Buttons
        btn_frame = tk.Frame(right, bg=BG_DARK)
        btn_frame.pack(fill="x")

        self._make_btn(btn_frame, "Collect Sample",
                       ACCENT, self._collect_sample).pack(fill="x", pady=2)
        self._make_btn(btn_frame, "Clear Transcript",
                       BG_CARD, self._clear_transcript).pack(fill="x", pady=2)
        self._make_btn(btn_frame, "Quit",
                       "#3d3d5c", self._quit).pack(fill="x", pady=(6, 2))

        hint = tk.Label(
            self.root,
            text="Use Collect Sample to teach the model a new gesture.",
            font=("Helvetica", 8),
            bg=BG_DARK, fg=TEXT_DIM,
        )
        hint.pack(pady=(0, 6))

        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    @staticmethod
    def _make_btn(parent, text, color, command):
        return tk.Button(
            parent, text=text,
            font=("Helvetica", 10, "bold"),
            bg=color, fg=TEXT_MAIN,
            activebackground=ACCENT, activeforeground="white",
            relief="flat", bd=0, padx=8, pady=6,
            cursor="hand2",
            command=command,
        )

    # Data/model init

    def _load_data(self):
        ensure_data_dir()
        self.samples, self.labels = load_samples()
        if self.samples is not None:
            n = len(self.labels)
            self.sample_count_var.set(f"Samples loaded: {n}")
            self.status_var.set(f"Loaded {n} training samples.")
        else:
            self.sample_count_var.set("Samples loaded: 0")
            self.status_var.set("No samples found. Using heuristics only.")

    def _init_camera_and_model(self):

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not open webcam."
            )
            self.root.destroy()
            return

        # MediaPipe model
        try:
            self.landmarker = create_landmarker()
        except FileNotFoundError as exc:
            messagebox.showerror("Model Not Found", str(exc))
            self.root.destroy()
            return

        self.status_var.set("Camera and model ready. Show your hand.")
        self.root.after(30, self._update_frame)

    # Main loop

    def _update_frame(self):
        
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Failed to read from camera.")
            self.root.after(500, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp_image.Image(
            image_format=mp_image.ImageFormat.SRGB, data=frame_rgb
        )
        result = self.landmarker.detect_for_video(mp_frame, self.timestamp_ms)
        self.timestamp_ms += int(1000 / 30)

        prediction = None
        method_tag = ""
        self.latest_landmarks = None
        self.latest_vector    = None

        if result and result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            self.latest_landmarks = hand_lm

            # Draw skeleton
            drawing_utils.draw_landmarks(
                frame,
                hand_lm,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_utils.DrawingSpec(color=(0, 255, 80),  thickness=2, circle_radius=3),
                drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            # Predict
            vec = normalize_landmarks(hand_lm)
            self.latest_vector = vec

            knn_pred = predict_from_samples(vec, self.samples, self.labels)
            if knn_pred is not None:
                prediction = knn_pred
                method_tag = "kNN from samples"
            else:
                prediction = heuristic_label(hand_lm)
                if prediction:
                    method_tag = "rule-based"

            # Overlay prediction
            label_text = prediction if prediction else "unknown"
            cv2.putText(
                frame, f"Sign: {label_text}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (20, 240, 20), 2, cv2.LINE_AA,
            )

        # Update prediction widgets
        if prediction:
            self.sign_var.set(prediction)
            self.method_var.set(method_tag)
        else:
            self.sign_var.set("—")
            self.method_var.set("no hand detected")

        # Append to transcript (once per second)
        now = time.time()
        if prediction and (now - self.last_append_time) >= 1.0:
            if prediction != self.last_prediction:
                self._append_transcript(prediction)
            self.last_prediction = prediction
            self.last_append_time = now

        display = cv2.resize(frame, (480, 360))
        img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self._update_frame)

    # Transcript helpers

    def _append_transcript(self, text: str):
        self.transcript_text.configure(state="normal")
        self.transcript_text.insert("end", text + " ")
        self.transcript_text.see("end")
        self.transcript_text.configure(state="disabled")

    def _clear_transcript(self):
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.configure(state="disabled")
        self.last_prediction = None

    # Sample collection

    def _collect_sample(self):

        if self._collect_pending:
            return  # dialog already open

        if self.latest_vector is None:
            messagebox.showwarning(
                "No Hand Detected",
                "Please show your hand clearly in front of the camera, then click Collect Sample again."
            )
            return

        self._collect_pending = True
        vec = self.latest_vector.copy()

        label = simpledialog.askstring(
            "Collect Sample",
            "Enter a label for this gesture (i.e. A, Hello, Yes):",
            parent=self.root,
        )
        self._collect_pending = False

        if not label or not label.strip():
            return
        label = label.strip()

        if self.samples is None or self.labels is None:
            self.samples = np.array([vec], dtype=np.float32)
            self.labels  = [label]
        else:
            if isinstance(self.samples, np.ndarray):
                sample_list = self.samples.tolist()
            else:
                sample_list = list(self.samples)
            sample_list.append(vec.tolist())
            self.samples = np.array(sample_list, dtype=np.float32)
            self.labels.append(label)

        save_samples(self.samples, self.labels)
        n = len(self.labels)
        self.sample_count_var.set(f"Samples loaded: {n}")
        self.status_var.set(f"Saved sample for '{label}'. Total: {n}")

    # Exit

    def _quit(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()

    style = ttk.Style(root)
    available = style.theme_names()
    for preferred in ("clam", "alt", "default"):
        if preferred in available:
            style.theme_use(preferred)
            break

    app = ASLTranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()