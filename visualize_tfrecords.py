import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TD_DIR = os.path.join(SCRIPT_DIR, 'training_dataset')

if TD_DIR not in sys.path:
    sys.path.insert(0, TD_DIR)

from visualize_tfrecords import *

if __name__ == '__main__':
    main()
