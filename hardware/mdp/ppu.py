import sys
import pydls
import pywrapstdvector as pysv
import struct
import time


def load_program(fn):
    rv = []
    with open(fn, 'r') as f:
        word = f.read(4)
        while word != '':
            rv.append(struct.unpack('>I', word)[0])
            word = f.read(4)
    return rv

def save_result(fn, chunk):
    with open(fn, 'w') as f:
        for word in chunk:
            f.write(struct.pack('>I', word))

def clear_memory(c):
    data = pysv.Vector_UInt32()
    data.extend([0 for i in range(4096)])
    pydls.omnibus_block_write(c, 0, data)

def start_program(c):
    ctrl = 0x00000001  # turn of reset
    pydls.omnibus_write(c, 0x00020000, ctrl)
    status = pydls.omnibus_read(c, 0x00020001)
    while status == 0:
        print "Status: ", status
        status = pydls.omnibus_read(c, 0x00020001)
        time.sleep(0.2)
    print "Status: ", status

    ctrl = 0 # turn on reset
    pydls.omnibus_write(c, 0x00020000, ctrl)

def compare_result(result, expected):
    rv = True
    for i in range(min(len(result), len(expected))):
        if result[i] != expected[i]:
            print "*** mismatch at %d: expected: %08x, actual: %08x" % (
                    i, expected[i], result[i])
            rv = False
    return rv

def write_program(c, prog, data=[]):
    img = pysv.Vector_UInt32()
    img.extend(prog)
    img.extend(data)
    pydls.omnibus_block_write(c, 0, img)

def run_program(c, prog, data):
    write_program(c, prog, data)
    start_program(c)


def read_mailbox(c, offset, end=4096):
    img = pydls.omnibus_block_read(c, offset, end - offset)
    return [ i for i in img ]

def write_mailbox(c, offset, data):
    img = pysv.Vector_UInt32()
    img.extend(data)
    pydls.omnibus_block_write(c, offset, img)

def show_mailbox(c):
    mbox = read_mailbox(c, 0x3000/4, 0x3000/4 + 3*8 + 1 + 400/4)
    col = 0
    rv = []
    for w in mbox:
        sys.stdout.write('%08x ' % (w))

        if col == 7:
            sys.stdout.write('\n')
            col = 0
        else:
            col += 1

        rv.extend([ (w >> (i*8)) & 0xff for i in [ 3, 2, 1, 0 ] ])

    sys.stdout.write('\n')

