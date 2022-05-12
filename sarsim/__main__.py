import argparse
import sys
import io

from sarsim import simscene, commands, simstate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMS SAR Simulator', prog='python3 -m sarsim')

    parser.add_argument('--gui', action='store_true', help='run an interactive GUI based on PyQt')
    parser.add_argument('--write-stubs', action='store_true', help='create stub files for better IDE autocompletion')
    parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use')
    parser.add_argument('--do-file', action='extend', nargs='+', type=argparse.FileType('r'), dest='do', help='run the script file(s) specified. See --help-scripting')
    parser.add_argument('--do', action='extend', nargs='+', help='run the script command specified. See --help-scripting')
    parser.add_argument('--help-scripting', action='store_true', help='show information about scripting')

    args = parser.parse_args()

    # Track if a valid command was executed to suppress warning if --gui was not selected.
    command_executed = False

    # Create global programm state
    state = commands.ProgramState(
        args=args,
        simstate=simstate.create_state(),
        scene=simscene.create_default_scene()
        )

    if args.write_stubs:
        print('Writing Stub-Files')
        simstate.write_simstate_stub_file()
        print('simstate stub-file written.')
        command_executed = True

    if args.help_scripting:
        print(commands.help_text)
        command_executed = True

    if args.do and len(args.do) > 0:
        for entry in args.do:
            if isinstance(entry, io.TextIOWrapper): # is a file, execute line by line
                for line in entry.readlines():
                    commands.run_command(line.strip(), state)
            else: # is string, execute directly
                commands.run_command(entry, state)

        #python3 -m sarsim --do-file script-test script-test --do a b --do c

        command_executed = True

    if args.gui:
        print('Launching SAR-Sim GUI.')

        from . import gui
        gui.run_gui(state)

        print('SAR-Sim GUI closed.')
        exit(0)

    else:
        if not command_executed:
            print('Warning: No operation specified. Try --gui or --help-scripting.\n', file=sys.stderr)
            parser.print_help(file=sys.stderr)
            exit(1)
        else:
            exit(0)
