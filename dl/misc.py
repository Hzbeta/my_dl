import logging

def getLogger(name, level=logging.DEBUG, fmt_str='[%(asctime)s][%(levelname).1s]%(message)s', date_fmt='%H:%M:%S'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(fmt_str, date_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)
    return logger