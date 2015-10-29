import logging

#logging.config.fileConfig('logging.conf')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

PREDICTIONS_DIR = "./.cache"