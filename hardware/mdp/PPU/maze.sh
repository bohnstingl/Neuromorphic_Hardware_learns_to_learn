waf configure install --target=maze
powerpc-eabi-objcopy -O binary bin/maze bin/maze.raw
srun -p dls --gres B291698 python run_program.py bin/maze.raw --data_in MailboxContent --as_string
