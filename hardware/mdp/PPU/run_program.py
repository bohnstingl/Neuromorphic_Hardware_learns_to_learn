import sys
import pydls
import argparse
import struct
import itertools
import pylogging
import pywrapstdvector as pysv

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute PPU programs")
    parser.add_argument('program', type=str)
    parser.add_argument('--board_id', type=str, default=None)
    parser.add_argument('--data_in', type=str)
    parser.add_argument('--data_out', type=argparse.FileType('wb'))
    parser.add_argument('--as_string', action='store_true')

    # No matter how long the program runs, it will always be stopped after
    # `wait` FPGA cycles. If the program finishes earlier, the PPU goes to
    # sleep mode while the FPGA still waits for this fixed amount of cycles.
    parser.add_argument(
            '--wait',
            type=int,
            default=int(1E7),
            help="Number of FPGA cycles after which the PPU is stopped")
    args = parser.parse_args()

    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')

    # Load the data
    program = pydls.Ppu_program()
    program.read_from_file(args.program)

    mailbox = pydls.Mailbox()
    if args.data_in is not None:
        load_mailbox(mailbox, args.data_in)

    # Setup synram control register
    # These are magic numbers which configure the timing how the synram is
    # written.
    synram_config_reg = pydls.Synram_config_reg()
    synram_config_reg.pc_conf(1)
    synram_config_reg.w_conf(1)
    synram_config_reg.wait_ctr_clear(1)

    # PPU control register
    ppu_control_reg_start = pydls.Ppu_control_reg()
    ppu_control_reg_start.inhibit_reset(True)

    ppu_control_reg_end = pydls.Ppu_control_reg()
    ppu_control_reg_end.inhibit_reset(False)

    # Playback memory program
    builder = pydls.Dls_program_builder()
    builder.set_synram_config_reg(synram_config_reg)
    builder.set_mailbox(mailbox)
    builder.set_ppu_program(program)
    builder.set_ppu_control_reg(ppu_control_reg_end)
    builder.set_ppu_control_reg(ppu_control_reg_start)
    builder.set_time(0)
    builder.wait_until(args.wait)
    status_handle = builder.get_ppu_status_reg()
    builder.set_ppu_control_reg(ppu_control_reg_end)
    mailbox_handle = builder.get_mailbox()
    builder.halt()

    # Connect
    if args.board_id is not None:
        connection = pydls.connect(args.board_id)
    else:
        board_ids = pydls.get_allocated_board_ids()
        assert(len(board_ids) == 1)
        assert(len(board_ids[0]) != 0)
        connection = pydls.connect(board_ids[0])
    pydls.soft_reset(connection)

    # Transfer execute and copy back results
    builder.transfer(connection, 0x0)
    builder.execute(connection, 0x0)
    builder.fetch(connection)

    # Disconnect
    connection.disconnect()

    # Read mailbox
    mailbox_result = mailbox_handle.get()
    if args.as_string:
        print_mailbox_string(mailbox_result)
    else:
        print_mailbox(mailbox_result)

    # Dump mailbox
    if args.data_out is not None:
        dump_mailbox(mailbox_result, args.data_out)

    # Check status register
    status_reg_result = status_handle.get()
    if status_reg_result.sleep() is not True:
        raise AssertionError("PPU Program did not finish")
