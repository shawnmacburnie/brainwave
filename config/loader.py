import modules.nn256 as nn256
import modules.nn1024 as nn1024
import modules.nn256_256 as nn256_256
import modules.nn1024_256 as nn1024_256
import modules.nn1024_1024 as nn1024_1024
import modules.nn256_256_256 as nn256_256_256
import modules.nn256_256_256_256 as nn256_256_256_256

def load(file):
    print 'Setting up network ' + file
    if file == 'nn256':
        return nn256
    if file == 'nn1024':
        return nn1024

    if file == 'nn256_256':
        return nn256_256
    if file == 'nn1024_256':
        return nn1024_256
    if file == 'nn1024_1024':
        return nn1024_1024

    if file == 'nn256_256_256':
        return nn256_256_256

    if file == 'nn256_256_256_256':
        return nn256_256_256_256
