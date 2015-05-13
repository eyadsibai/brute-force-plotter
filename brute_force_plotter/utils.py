import logging
import os
import errno

import matplotlib.pyplot as plt


def ignore_if_exist_or_save(func):
    def wrapper(*args, **kwargs):

        file_name = kwargs['file_name']

        if os.path.isfile(file_name):
            plt.close('all')
        else:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)
            plt.gcf().savefig(file_name, dpi=120)
            plt.close('all')

    return wrapper


def make_sure_path_exists(path):
    logging.debug('Make sure {} exists'.format(path))
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            return False
    return True
