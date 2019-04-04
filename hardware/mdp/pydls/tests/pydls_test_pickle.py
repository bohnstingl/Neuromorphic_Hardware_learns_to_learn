#!/usr/bin/env python
import unittest
import copy
import inspect
import pickle
import pydls

class PyDLS_test(unittest.TestCase):
    @classmethod
    def main(self, *args, **kwargs):
        unittest.main(*args, **kwargs)

pydls_classes = {}
for name, obj in inspect.getmembers(pydls):
    if inspect.isclass(obj):
        pydls_classes[name] = obj


class Test_PyDLS_Bindings(PyDLS_test):
    pass

excluded_in_generate = [
    # manually excluded via generate.py
    'Ppu_program_on_dls',
    'Ppu_on_dls',
    'Mailbox_on_dls',
    'Dls_program_builder',
    'Connection',
    # enums...
    'Dac',
    'Spi',
]

def add_test_has_json():
    for name, obj in pydls_classes.items():
        def generate_test_function(obj):
            def test_function(self):
                if (obj.__name__.startswith('Vector')
                    or obj.__name__.startswith('Array')
                    or obj.__name__.startswith('_std_')
                    or obj.__name__ == 'int'
                    or obj.__name__ in excluded_in_generate
                ):
                    raise unittest.SkipTest(
                        'pickeling unsupported: %s' % name)
                self.assertTrue(hasattr(obj, "to_json"))
                self.assertTrue(hasattr(obj, "from_json"))
            return test_function
        func = generate_test_function(obj)
        func.__name__ = 'test_has_json_%s' % name
        setattr(Test_PyDLS_Bindings, func.__name__, func)
add_test_has_json()

# FIXME: no default constructors
not_default_constructible = [
    'Synapse_row',
    'Synapse_driver',
    'Synapse_column',
    'Neuron_index',
    'Dac_channel',
    'Cap_mem_row',
    'Cap_mem_column',
    'Cadc_channel',
    'Address_on_mailbox',
]

def add_test_default_constructible():
    for name, obj in pydls_classes.items():
        def generate_test_function(obj):
            def test_function(self):
                if obj.__name__ in not_default_constructible:
                    raise unittest.SkipTest(
                        'default construction unsupported: %s' % name)
                obj()
            return test_function
        func = generate_test_function(obj)
        func.__name__ = 'test_pickle_%s' % name
        setattr(Test_PyDLS_Bindings, func.__name__, func)
add_test_default_constructible()



def add_test_pickle():
    for name, obj in pydls_classes.items():
        def generate_test_function(obj):
            def test_function(self):
                if (obj.__name__.startswith('_std_')
                    or obj.__name__.startswith('Vector')
                    or obj.__name__ in not_default_constructible
                    or obj.__name__ in excluded_in_generate
                ):
                    raise unittest.SkipTest(
                        'pickeling unsupported: %s' % name)
                a = obj()
                dumpstr = pickle.dumps(a)
                b = pickle.loads(dumpstr)
                # FIXME: almots all operator== missing... self.assertEqual(a, b)
            return test_function
        func = generate_test_function(obj)
        func.__name__ = 'test_pickle_%s' % name
        setattr(Test_PyDLS_Bindings, func.__name__, func)
add_test_pickle()


if __name__ == '__main__':
    # etc...
    Test_PyDLS_Bindings.main()
