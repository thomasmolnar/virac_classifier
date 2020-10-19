import configparser
import getpass

class configuration(object):
    def __init__(self):
        self.config_ = configparser.ConfigParser()
        self.config_.read_file(open('config.cfg'))
        self.wsdb_kwargs = dict(self.config_['wsdb'])
        self.wsdb_kwargs.update({
            'asDict':True,
            'preamb':'set enable_seqscan to off; set enable_mergejoin to off; set enable_hashjoin to off;',
            'db':'wsdb'})
    def __getitem__(self, l):
        return self.config_['general'][l]
    def __setitem__(self, l, v):
        self.config_['general'][l]=str(v)
    def request_password(self):
        self.wsdb_kwargs['password'] = getpass.getpass()
