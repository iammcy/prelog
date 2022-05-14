import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


class prelog():

    def __init__(self, exp_name: str):
        
        '''
            exp_name: 实验名称
        '''
        
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = exp_name

        self.init_dir(self.exp_dir)
        self.init_logger(self.exp_name)
        self.init_writer(self.exp_name)


    def init_dir(self, exp_dir: str):
        # state
        self.state_dir = os.path.join(exp_dir, "state")
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)

        # tensorboard log
        self.tb_dir = os.path.join(exp_dir, "tb")
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir)

        # logging
        self.log_dir = os.path.join(exp_dir, "log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    def init_logger(self, file_name: str):
        
        '''
            file_name: 日志文件名
        '''

        log_file = os.path.join(self.log_dir, file_name + '.log')

        logging.basicConfig(
            format='%(asctime)s | %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            filename=log_file,
            filemode='a+'
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        self.logging = logging

    
    def init_writer(self, exp_name: str):
        self.writer = SummaryWriter(os.path.join(self.tb_dir, exp_name))


    def save_model(self, model):
        torch.save(model.state_dict(), self.exp_name + ".pth")

    