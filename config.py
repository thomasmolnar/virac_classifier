import os
import configparser
import getpass

thispath=os.path.dirname(os.path.abspath(__file__))+'/'

class configuration(object):
    def __init__(self):
        self.config_ = configparser.ConfigParser()
        self.config_.read_file(open(thispath+'config.cfg'))
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
