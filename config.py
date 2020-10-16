import configparser
import getpass

class configclass(object):
    def __init__(self):
        self.config_ = configparser.ConfigParser()
        self.config_.read_file(open('config.cfg'))
    def __getitem__(self, l):
        return self.config_['general'][l]

config = configclass()
    
def request_password():
    config['password'] = getpass.getpass()