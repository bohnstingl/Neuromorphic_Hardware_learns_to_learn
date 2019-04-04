import sys
import pydls
import argparse
import struct
import itertools
import pylogging
import pywrapstdvector as pysv
import numpy as np

def load_mailbox(mailbox, filename):
    words = []
    with open(filename, 'rb') as f:
        while True:
            d = f.read(4)
            if len(d) == 0:
                break
            d = d + '\x00' * (4 - len(d))
            word = struct.unpack('>L', d)[0]
            words.append(word)
    for index, word in enumerate(words):
        mailbox.set_word(pydls.Address_on_mailbox(index), word)

def load_mailbox_from_virtual_file(mailbox, data):
    words = []
    index = 0
    while True:
        if len(data) - index == 0:
            break
        
        if len(data) - index > 4:
            d = data[index:index+4]
            index += 4
        else:
            d = data[index:len(data)]
            d = d + '\x00' * (4 - len(d))
            index = len(data)

        word = struct.unpack('>L', d)[0]
        words.append(word)

    for index, word in enumerate(words):
        mailbox.set_word(pydls.Address_on_mailbox(index), word)

def bytes_of_mailbox(mailbox):
    words = mailbox.export_words()
    bytes_in_words = (struct.unpack('BBBB', struct.pack('>I', word)) for word in words)
    return itertools.chain.from_iterable(bytes_in_words)

def print_mailbox_string(mailbox):
    bytes_gen = bytes_of_mailbox(mailbox)
    # Write characters in the mailbox up to the first null byte
    for byte in itertools.takewhile(lambda x : x != 0, bytes_gen):
        sys.stdout.write("{:c}".format(byte))
    sys.stdout.write("\n")

def print_mailbox(mailbox, chunk_size=16):
    bytes_gen = bytes_of_mailbox(mailbox)
    for index, byte in enumerate(bytes_gen):
        if (index % chunk_size) == 0:
            sys.stdout.write("\n{:04x} | ".format(index))
        sys.stdout.write("{:02x} ".format(byte))
    sys.stdout.write("\n")

def dump_mailbox(mailbox, file_handle):
    for word in mailbox.export_words():
        file_handle.write(struct.pack('>I', word))

def readRange_mailbox(mailbox, lowerAddress, upperAddress):
    byteList = []
    bytes_gen = bytes_of_mailbox(mailbox)
    for index, byte in enumerate(bytes_gen):
        if index >= lowerAddress and index < upperAddress:        
            byteList.append(byte)

    return byteList

def convertByteListToInt(byteList, sign):

    if len(byteList) % 4 != 0:
        raise Exception('given array has not correct length: ' + str(len(byteList)))
    
    intList = []
    for i in range(0, len(byteList), 4):
        num = (byteList[i] << 24) + (byteList[i + 1] << 16) + (byteList[i + 2] << 8) + (byteList[i + 3])
        if sign:        
            intList.append(np.int32(num))
        else:
            intList.append(np.uint32(num))
    return intList

def convertByteListToInt8(byteList, sign):
    if sign:
        return struct.unpack('%ub' % len(byteList), bytearray(byteList))
    else:
        return struct.unpack('%uB' % len(byteList), bytearray(byteList))
