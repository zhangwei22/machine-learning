import time

from util.mysql_util import *

if __name__ == '__main__':
    err_rate = 0.36333

    bean_data = {}
    bean_data['data_sets'] = 2000
    bean_data['origin_sets'] = 5000
    bean_data['algorithm'] = 1
    bean_data['learning_rate'] = 0.0
    bean_data['iterations'] = 2000
    bean_data['handle_bc_data'] = 0
    bean_data['error_rate'] = err_rate
    bean_data['correct_rate'] = 1 - err_rate
    bean_data['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    bean_data['operator'] = 'master'

    print(bean_data)
    add(bean_data)