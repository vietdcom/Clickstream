import tkinter as tk
import csv
import time
import os
import psutil
import win32gui
import win32process
from threading import Thread
from pynput import keyboard

LOG_FILE = "user_logs.csv"
APP_CATEGORIES = {
    "epicgameslauncher": "gaming",
    "eaplay": "gaming",
    "discord": "chatting",
    "zalo": "chatting",
    "chrome": "browsing",
    "opera": "browsing",
    "firefox": "browsing",
    "edge": "browsing",
    "Visual Studio Code": "coding",
    "pycharm": "coding",
    "intellij": "coding",
    "notepad++": "coding",
    "code": "coding",
    "word": "working",
    "excel": "working",
    "powerpnt": "working",
    "facebook":"entertainment",
    "spotify": "entertainment",
    "netflix": "entertainment",
    "YouTube": "entertainment",
}

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "event_type", "description", "type"])

def log_event(event_type, description, type):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_type, description, type])
    print(f"[{timestamp}] {event_type}: {description}, {type}")

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
        label = classify_app(app)
        if (app != last_app or title != last_title) and app != "python.exe":
            log_event("active_app", f"{app} | {title} ", f"{label}")
            last_app = app
            last_title = title
        time.sleep(5)

def on_press(key):
    try:
        k = key.char if hasattr(key, 'char') else str(key)
    except Exception:
        k = str(key)
    app, title = get_active_window_info()
    label = classify_app(app)
    log_event("keypress", f"type something | {app} | {title} ", f"{label}")

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

def start_gui():
    root = tk.Tk()
    root.title("Auto App Logger")

    tk.Label(root, text="Nhấn các nút để mô phỏng hoạt động:").pack(pady=5)

    actions = [
        ("Lướt Web", "browsing"),
        ("Viết Code", "coding"),
        ("Chơi Game", "gaming"),
        ("Làm Việc", "working"),
        ("Mua Sắm", "shopping"),
    ]

    for label, action in actions:
        tk.Button(root, text=label, width=30,
                  command=lambda act=action: log_event("click", act)).pack(pady=2)

    entry = tk.Entry(root, width=40)
    entry.pack(pady=10)
    entry.bind("<Key>", lambda e: log_event("input", "User typed something"))
    entry.insert(0, "Nhập gì đó vào đây...")

    root.mainloop()

Thread(target=auto_log_running_app, daemon=True).start()
start_keyboard_listener()
start_gui()
