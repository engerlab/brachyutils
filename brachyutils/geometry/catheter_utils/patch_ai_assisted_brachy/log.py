import os
from typing import List


class Logger:
    def __init__(self, log_folder:str=None, log_file_name:str="log.txt"):
        self.log_folder = log_folder
        self.log_file = os.path.join(log_folder, log_file_name)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            
    def log(self, message:List[str]):
        assert isinstance(message, list)
        with open(self.log_file, "a") as f:
            f.write("\n".join(message))
