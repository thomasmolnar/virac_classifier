import configparser
import getpass

config = configparser.ConfigParser()
config.read_file(open('config.cfg'))

def request_password():
    config['password'] = getpass.getpass()