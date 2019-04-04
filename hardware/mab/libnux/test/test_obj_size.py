#!/usr/bin/env python

import argparse
import sys
from junit_xml import TestSuite, TestCase
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_abspath")
    parser.add_argument("xmlfile_abspath")
    parser.add_argument("max_size", type=int)
    args = parser.parse_args()

    # extract the heap_base symbol from the linked file; that's the first
    # address after the linked data area, i.e. the program size
    cmd = 'nm -td {} | sed -n -n \'s/\([^ ]\+\) T heap_base$/\\1/p\''.format(
        args.test_abspath)
    out = subprocess.check_output(cmd, shell=True).strip()
    size = int(out)

    fail = False
    if size > args.max_size:
        fail = True

    test_case = TestCase('test_obj_size', args.test_abspath, 1, size, '')
    if fail:
        test_case.add_failure_info(
            message='heap start after byte {}: {}'.format(args.max_size, size))
    ts = [TestSuite("libnux", [test_case])]

    with open(args.xmlfile_abspath, 'w') as fd:
        TestSuite.to_file(fd, ts, prettyprint=True)

    sys.exit(fail)
