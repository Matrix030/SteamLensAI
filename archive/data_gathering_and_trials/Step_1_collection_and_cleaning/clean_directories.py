#this program removes the directory with no data.

import os

def clean_directories(base_path, log_file='non_empty_dirs.txt', action_log='log.txt', dry_run=True):

    with open(log_file, 'w') as non_empty_log, open(action_log, 'w') as action_log_f:
        for root, dirs, files in os.walk(base_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        action_log_f.write(f"Empty: {dir_path}\n")
                        if not dry_run:
                            os.rmdir(dir_path)
                            print(f"Deleted: {dir_path}")
                        else:
                            print(f"[DRY RUN] Would delete: {dir_path}")
                    else:
                        non_empty_log.write(dir_path + '\n')
                        action_log_f.write(f"Non-empty: {dir_path}\n")
                except Exception as e:
                    error_msg = f"Error processing {dir_path}: {e}"
                    print(error_msg)
                    action_log_f.write(error_msg + '\n')

# === SAFETY FIRST ===
base_path = '../app_details_json'
print(f"You're about to scan and clean: {base_path}")
proceed = input("Proceed? This may delete empty folders. Type 'YES' to continue: ")

if proceed == 'YES':
    clean_directories(base_path, dry_run=False)  
else:
    print("Operation cancelled.")
