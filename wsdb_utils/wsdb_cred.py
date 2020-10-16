### WSDB credentials with password to be included

wsdb_kwargs = {'host':'cappc127',
               'user':'thomas_molnar',
#                'password':'',
               'asDict':True,
               'preamb':
                'set enable_seqscan to off; set enable_mergejoin to off; set enable_hashjoin to off;',
               'db':'wsdb'}