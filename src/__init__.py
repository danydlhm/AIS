from logging import getLogger, StreamHandler, DEBUG
from logging.handlers import RotatingFileHandler
from sys import stdout
from os.path import join,exists
from os import makedirs

logger = getLogger('AIS')
logger.setLevel(DEBUG)
logs_path = 'logs'
if not exists(logs_path):
    makedirs(logs_path, exist_ok=True)
logger.addHandler(RotatingFileHandler(join(logs_path,'ais.log'), encoding='utf-8'))
logger.addHandler(StreamHandler(stdout))