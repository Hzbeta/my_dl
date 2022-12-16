import logging, os
from datetime import datetime


def get_logger(name,
               level=logging.DEBUG,
               log_dir=None,
               log_time=None,
               fmt_str='[%(asctime)s][%(levelname).1s]%(message)s',
               date_fmt='%H:%M:%S'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(fmt_str, date_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)
    
    if(log_dir and log_time):
        os.makedirs(log_dir, exist_ok=True)
        #日志文件名为当前时间
        log_file_name = log_time + '.log'
        log_file_path = os.path.join(log_dir, log_file_name)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_time():
    #获取当前日期和时间，包含毫秒
    return datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S:%f')[:-3]