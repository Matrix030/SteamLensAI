# This programs reads the log details to kill a process when a certain string appears in the log file.
import time
import psutil
import os

# Config
log_file_path = r"../app_details_json/steam_app_scraper.log"  # Replace with actual path
target_phrase = "Processing appID:"


def get_pid_by_name(name):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if name.lower() in (proc.info['name'] or "").lower() or \
               any(name.lower() in cmd.lower() for cmd in cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def monitor_log_and_kill(target_pid):
    print(f"Watching log: {log_file_path}")
    try:
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Move to the end of the file
            f.seek(0, os.SEEK_END)

            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue

                print("LOG:", line.strip())
                if target_phrase in line:
                    print(f"Found '{target_phrase}' – killing process {target_pid}")
                    try:
                        psutil.Process(target_pid).terminate()
                        print(f"Process {target_pid} terminated.")
                    except psutil.NoSuchProcess:
                        print(f"Process {target_pid} not found.")
                    break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pid = get_pid_by_name("getAppDetails.py")  # Replace with your target process keyword
    if pid:
        print(f"Found PID: {pid}")
    else:
        print("Process not found.")
    monitor_log_and_kill(pid)