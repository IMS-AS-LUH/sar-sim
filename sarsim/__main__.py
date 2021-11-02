import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMS SAR Simulator', prog='python3 -m sarsim')

    parser.add_argument('--gui', action='store_true', help='run an interactive GUI based on PyQt')
    parser.add_argument('--write-stubs', action='store_true', help='create stub files for bedder IDE autocompletion')
    parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use')

    args = parser.parse_args()

    # Track if a valid command was executed to suppress warning if --gui was not selected.
    command_executed = False

    if args.write_stubs:
        print('Writing Stub-Files')
        from . import simstate
        simstate.write_simstate_stub_file()
        print('simstate stub-file written.')
        command_executed = True

    if args.gui:
        print('Launching SAR-Sim GUI.')

        from . import gui
        gui.run_gui(args)

        print('SAR-Sim GUI closed.')
        exit(0)

    else:
        if not command_executed:
            print('Error: CLI Mode not implemented yet.\n', file=sys.stderr)
            parser.print_help(file=sys.stderr)
            exit(1)
        else:
            exit(0)
