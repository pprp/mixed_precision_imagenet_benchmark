from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TensorboardLogger():
    def __init__(self, log_dir="tfboard_logs"):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        self.epoch = 0
        self.mode = ''
        self.timer = datetime.now()

    def add_scalar(self, title, name, item):
        self.writer.add_scalar(title+'/'+name, item)
    