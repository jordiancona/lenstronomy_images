import lenstronomy as ln
import numpy as np

class lens:
    def __init__(self, total_images = 1000, model = 'SIE') -> None:
        self.total_images = total_images
        self.model = model
    

