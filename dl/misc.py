import logging, os, time


def getLogger(name, log_dir, level=logging.DEBUG, fmt_str='[%(asctime)s][%(levelname).1s]%(message)s', date_fmt='%H:%M:%S'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(fmt_str, date_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    #日志文件名为当前时间
    log_file_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '.log'
    log_file_path = os.path.join(log_dir, log_file_name)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger