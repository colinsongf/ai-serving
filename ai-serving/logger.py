# -*- coding: utf-8 -*-

import logging


def get_logger():
    logger = logging.getLogger("AiServing")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger


LOGGER = get_logger()
