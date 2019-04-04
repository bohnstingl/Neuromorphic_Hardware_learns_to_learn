#!/usr/bin/env python

from pyplusplus.module_builder import call_policies
from pywrap import containers, namespaces, matchers, classes
from pywrap.namespace_util import NamespaceUtil
from pywrap.wrapper import Wrapper
from pyplusplus import decl_wrappers
from pygccxml.declarations import templates

wrap = Wrapper()
mb = wrap.mb
ns_util = NamespaceUtil()

#mb.code_creator.add_include("rw_api/flyspi_com.h")

# include everything from frickel_dls namespace
n = mb.namespace("frickel_dls")
n.include()
for c in n.classes(allow_empty=True):
    c.include_files.append("rw_api/flyspi_com.h")
for c in n.free_functions(allow_empty=True):
    c.include_files.append("rw_api/flyspi_com.h")

n_uni = mb.namespace('uni')
for c in n_uni.classes(allow_empty=True):
    if c.name == 'Spike':
        c.include_files.append("rw_api/flyspi_com.h")
        c.include()


# Special fix up
containers.extend_std_containers(mb)
namespaces.include_default_copy_constructors(mb)



# return value policies
#ref_funcs = [ mb.namespace("frickel_dls").free_function("connect") ]
#for f in ref_funcs:
    #f.call_policies = call_policies.custom_call_policies("bp::return_value_policy<bp::reference_existing_object>")

# Try to capture the std::container dependcies
std_filter = matchers.and_matcher_t([
    matchers.declaration_matcher_t(decl_type=decl_wrappers.class_t),
    matchers.namespace_contains_matcher_t("std")
])

for ns in [ n, n_uni ]:
    for decl in namespaces.get_deps(ns, matchers.namespace_contains_matcher_t("std")):
        if decl.indexing_suite or decl.name.startswith("bitset"):
            decl.include()

ns_util.add_namespace(mb.namespace("std"))

def get_fops(name="operator<<"):
    return [
        tc.decl_string
        for tc in [op.get_target_class()
            for op in mb.free_operators() if op.name == name]
        if tc]

# classes with operator<< defined
cls_w_ostream = get_fops()

def add_print_operator(c):
    """
    class exclusion helper
    because operator<< is missing in most classes

    Args:
        c: class
    Returns:
        True if __str__() should be wrapped
        False otherwise
    """
    return (c.decl_string in cls_w_ostream)

# add str() function
for ns in [n, n_uni]:
    for c in ns.classes(allow_empty=True):
        if add_print_operator(c):
            c.add_registration_code('def(bp::self_ns::str(bp::self_ns::self))')

for ns in [n]:
    for cls in ns.classes(allow_empty=True):
        # exclude `*_handle`s and some other classes from serialization
        if cls.name in [
            'Dls_program_builder',
            'Ppu_program_on_dls',
            'Connection',
            'Ppu_on_dls',
            'Mailbox_on_dls',
        ] or cls.name.endswith('_handle'):
            continue
        classes.add_pickle_suite(cls, serialization_framework='cereal')
        classes.add_to_and_from_json(cls)

# Spike in uni namespace...
cls = n_uni.class_('Spike')
classes.add_pickle_suite(cls, serialization_framework='cereal')
classes.add_to_and_from_json(cls)

# expose only public interfaces
namespaces.exclude_by_access_type(mb, ['variables', 'calldefs', 'classes'], 'private')
namespaces.exclude_by_access_type(mb, ['variables', 'calldefs', 'classes'], 'protected')

# exclude names begining with a single underscore or ending with Cpp
namespaces.exclude_by_regex(mb, ['calldefs'], r'(^_[^_])|(.*Cpp$)|(^impl$)')

# add context semantics to python-wrapping of Connection
classes.add_context_manager(n.class_("Connection"), n.class_("Connection").member_function("disconnect"))

wrap.finish()
