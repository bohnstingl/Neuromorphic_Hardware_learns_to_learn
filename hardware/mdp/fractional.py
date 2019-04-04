import numpy as np


def to_fractional(floating_number, precision=8):
    '''
    Converts a float to fractional representation
    floating number has to be in the interval [-1, 1)
    '''
    if not (-1 <= floating_number <= 1):
        raise ValueError('has to be in correct interval')

    bits = np.zeros(precision, dtype=np.int)
    if floating_number < 0:
        bits[0] = 1
        floating_number += 1

    ind = 1
    while ind < precision:
        if floating_number > .5:
            bits[ind] = 1
            floating_number -= .5
        ind += 1
        floating_number *= 2

    num = 0
    for bit in bits:
        num = num << 1
        num += int(bit)
    return num

def to_float(fractional_number, precision=8):
    '''
    Converts a fractional to a float representation
    '''
    
    if type(fractional_number) == tuple:
        fractional_number = list(fractional_number)
    if type(fractional_number) != list:
        fractional_number = [fractional_number]
    
    floatReturn = []
    for number in fractional_number:
        num = 0
        
        if (number >> (precision - 1)) == 1:
            num = -1

        for i in range(0, precision-1):
            if (((number >> i) << 7) % 256) == 128:
                num += 2**(-((precision-1)-i))

        floatReturn.append(num)

    return floatReturn

def test_fractional():
    fs = np.linspace(-1, 0.9999, 100)
    for f in fs:
        precision = 6
        num = to_fractional(f, precision)
        f_a = 0
        if num & (1 << (precision - 1)) != 0:
            f_a = -1
        for i in range(1, precision):
            if num & (1 << (precision - i - 1)) != 0:
                f_a += 2**(-i)
        if f - f_a > 2**(-precision + 1):
            raise Exception('wrong')
        print('{:08b}'.format(num))


if __name__ == '__main__':
    
    to_float(2)
