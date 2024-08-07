import logging
import json

def configure_logging(config):

    log_level = config['logging_level'].upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["log_path"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)
