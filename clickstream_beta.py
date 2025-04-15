import tkinter as tk
import csv
import time
import os
import psutil
import win32gui
import win32process
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from threading import Thread
from pynput import keyboard
import joblib
from datetime import datetime
from collections import defaultdict

LOG_FILE = "logs.csv"
MODEL_FILE = "clickstream_model_beta(5).pth"
ENCODER_FILE = "label_encoder(5).pkl"
SESSION_TIMEOUT = 60  # seconds
PREDICTION_FILE = "predictions.csv"
TOKEN2IDX_FILE = "token2idx(4).pkl"
token2idx = joblib.load(TOKEN2IDX_FILE)
vocab_size = len(token2idx) + 1

# === MODEL ===
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ft = nn.Flatten()
        self.lora = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Softplus(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        out = self.bn(hn[-1])
        out = self.ft(out)
        out = self.lora(out)
        out = self.fc(out)
        return out

# === Load model và encoder ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_classes=4, dropout=0.3).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()
label_encoder = joblib.load(ENCODER_FILE)

# === Logging ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "event_type", "description"])

def log_event(event_type, description):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_type, description])
    print(f"[{timestamp}] {event_type}: {description}")

def classify_app(app_name: str) -> str:
    app_name = app_name.lower().strip()
    for key, label in APP_CATEGORIES.items():
        if key in app_name:
            return label
    return "other"

def get_active_window_info():
    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        process_name = process.name()
        return process_name, window_title
    except Exception:
        return "Unknown", "Unknown"

def auto_log_running_app():
    last_app = ""
    last_title = ""
    while True:
        app, title = get_active_window_info()
        # label = classify_app(app)
        if (app != last_app or title != last_title) and app != "python.exe":
            log_event("active_app", f"{app} | {title}")
            last_app = app
            last_title = title
        time.sleep(5)

def on_press(key):
    try:
        k = key.char if hasattr(key, 'char') else str(key)
    except Exception:
        k = str(key)
    app, title = get_active_window_info()
    # label = classify_app(app)
    log_event("keypress", f"type | {app} | {title}")

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

# === Dự đoán hành vi từ file log ===
def predict_from_log():
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)

    sessions = []
    session_times = []
    current_session = []
    current_times = []
    last_time = None

    for _, row in df.iterrows():
        time = row['timestamp']
        token = f"{row['description']}|{row['event_type']}"
        token_id = token2idx.get(token, 0)  # 0 nếu token chưa gặp

        if last_time is None or (time - last_time).total_seconds() <= SESSION_TIMEOUT:
            current_session.append(token_id)
            current_times.append(time)
        else:
            if current_session:
                sessions.append(current_session)
                session_times.append((current_times[0], current_times[-1]))
            current_session = [token_id]
            current_times = [time]
        last_time = time

    if current_session:
        sessions.append(current_session)
        session_times.append((current_times[0], current_times[-1]))

    if not sessions:
        return

    sequences = [torch.tensor(s, dtype=torch.long) for s in sessions]
    lengths = torch.tensor([len(s) for s in sequences])
    # Sắp xếp giảm dần theo lengths
    sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
    sorted_sequences = [sequences[i] for i in sorted_idx]
    padded = pad_sequence(sorted_sequences, batch_first=True).to(device)


    with torch.no_grad():
        outputs = model(padded, sorted_lengths.to(device))
        preds = torch.argmax(outputs, dim=1)
        decoded = label_encoder.inverse_transform(preds.cpu().numpy())

    # Ghi kết quả vào file
    with open(PREDICTION_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["session_id", "timestamp_start", "timestamp_end", "predicted_activity"])
        for i, label in enumerate(decoded):
            start_time = session_times[i][0].strftime("%Y-%m-%d %H:%M:%S")
            end_time = session_times[i][1].strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([i+1, start_time, end_time, label])

    print("✅ Đã ghi kết quả dự đoán vào predictions.csv")

# === GUI ===
def start_gui():
    root = tk.Tk()
    root.title("Clickstream Activity Logger Beta")

    tk.Label(root, text="Nhấn nút mô phỏng hoặc dùng máy như bình thường.").pack(pady=5)

    actions = [
        ("Lướt Web", "browsing"),
        ("Viết Code", "coding"),
        ("Chơi Game", "gaming"),
        ("Làm Việc", "working"),
        ("Mua Sắm", "shopping"),
    ]

    for label, action in actions:
        tk.Button(root, text=label, width=30,
                  command=lambda act=action: log_event("click", act, act)).pack(pady=2)

    # tk.Button(root, text="📊 Dự đoán hoạt động từ log", bg="lightblue", command=predict_from_log).pack(pady=10)

    entry = tk.Entry(root, width=40)
    entry.pack(pady=10)
    entry.bind("<Key>", lambda e: log_event("input", "typing", "typing"))
    entry.insert(0, "Gõ gì đó vào đây...")

    root.mainloop()

def auto_predict():
    while True:
        predict_from_log()
        time.sleep(10)  # Dự đoán mỗi 10 giây


Thread(target=auto_log_running_app, daemon=True).start()
Thread(target=auto_predict, daemon=True).start()
start_keyboard_listener()
start_gui()
