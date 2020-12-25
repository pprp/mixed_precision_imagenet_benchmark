from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random 


class TensorboardLogger():
    def __init__(self, log_dir="tfboard_logs"):
        self.writer = SummaryWriter(log_dir, comment=str(random.randint(0,10000)))
        self.step = 0
        self.epoch = 0
        self.mode = ''
        self.timer = datetime.now()

    def add_scalar(self, title, name, scalar_value, global_step):
        self.writer.add_scalar(title+'/'+name, scalar_value, global_step)
