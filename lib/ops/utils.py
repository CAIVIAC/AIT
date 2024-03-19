import os
import datetime
from termcolor import cprint, colored

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def color(text, txt_color='green', attrs=['bold']):
    return colored(text, txt_color, attrs=attrs)

def printer(
    bold_info=None, bold_color='magenta',\
    prnt_info=None, log_level='INFO'
    ):

    log({'bold': bold_info,\
         'cbold': bold_color,\
         'info': prnt_info,\
         }, log_level=log_level)

def log(str_dict, log_level='LOG'):
    LEVEL_DICT = {'LOG': 'blue', 'INFO': 'green'}
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cprint('[', color=None, end='')
    cprint('{now}'.format(now=now), color='white', attrs=['bold'], end='')
    cprint('][', color=None, end='')
    cprint('{level}'.format(level='%-4s'%(log_level)),\
            color=LEVEL_DICT[log_level], attrs=['bold'], end='')
    cprint(']{space}'.format(space=' '*2), color=None, end='')
    if str_dict['bold']:
        cprint('{string}'.format(string=str_dict['bold']), color=str_dict['cbold'], attrs=['bold'], end='')

    if str_dict['info']:
        cprint('{string}'.format(string=str_dict['info']), color=None)
    else:
        cprint('{string}'.format(string=''), color=None)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
