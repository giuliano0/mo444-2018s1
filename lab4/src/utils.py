from __future__ import print_function

import os
import sys

def ask_user(question, default='yes'):
    valid = {
        'yes': True,
        'ye': True,
        'y': True,
        'no': False,
        'n': False
    }

    default = default.lower()

    if default is None:
        prompt = ' [y/n] (defaults to %s): ' % (default)
    elif default == 'yes':
        prompt = ' [Y/n]: '
    elif default == 'no':
        prompt = ' [y/N]: '
    else:
        raise ValueError('Invalid default answer. Valid values are "yes" or "no".')
    
    while True:
        #sys.stdout.write(question + prompt)
        print(question + prompt)
        choice = input().lower()

        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            #sys.stdout.write('Please answer either yes/y or no/n.')
            print('Please answer either yes/y or no/n.')
