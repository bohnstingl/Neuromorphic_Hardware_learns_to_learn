def options(opt):
    opt.load('nux_compiler')

def configure(conf):
    conf.load('nux_compiler')

def build(bld):
    bld(
        target = 'nux_inc',
        export_includes = ['.'],
    )

    bld.stlib(
        target = 'nux',
        source = [
                'src/exp.c',
                'src/fxv.c',
                'src/mailbox.c',
                'src/unittest.c',
                'src/unittest_mailbox.c',
                ],
        use = ['nux_inc'],
    )

    bld(
        name = 'nux_runtime',
        target = 'crt.o',
        source = ['src/crt.s'],
        use = ['asm'],
    )

    bld.program(
        target = 'test_unittest',
        source = ['test/test_unittest.c'],
        use = ['nux', 'nux_runtime'],
    )

    bld.program(
        target = 'test_vector',
        source = ['test/test_vector.c'],
        use = ['nux', 'nux_runtime'],
    )

    bld.program(
        target = "test_fxvsel",
        source = ["test/test_fxvsel.c"],
        use = ["nux", "nux_runtime"],
    )
