import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
gui_app_path = os.path.join(SCRIPT_DIR, 'training_dataset', 'gui_app.py')

if __name__ == '__main__':
    os.execv(sys.executable, [sys.executable, gui_app_path] + sys.argv[1:])
