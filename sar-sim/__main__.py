import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMS SAR Simulator', prog='python3 -m sar-sim')

    parser.add_argument('--gui', action='store_true', help='run an interactive GUI based on PyQt')

    args = parser.parse_args()

    if args.gui:
        print('Launching SAR-Sim GUI.')

        from . import gui
        gui.run_gui()

        print('SAR-Sim GUI closed.')
        exit(0)

    else:
        print('Error: CLI Mode not implemented yet.\n', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(1)
