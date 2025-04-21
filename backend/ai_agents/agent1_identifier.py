import numpy as np
class Agent1Identifier:
    def __init__(self, class_mapping):
        self.class_to_idx = class_mapping
        self.idx_to_class = {v: k for k, v in class_mapping.items()}

    def identify(self, prediction):
        if isinstance(prediction, (list, np.ndarray)):
            return self.idx_to_class.get(np.argmax(prediction), "Unknown")
        return self.idx_to_class.get(int(prediction), "Unknown")
