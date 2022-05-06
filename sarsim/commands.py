# This module contains simple commands that may be called from scripts (and are used by the GUI)

from typing import Callable

from datetime import datetime

from . import simstate

help_text = """
Information on scripting support.

There are two types of commands: Assignments and actions.
An assignment has the form "name=value", while a action consists only of the
name of the action "print_date".

Assignments can be used to change parameters in the current simulator state. There must
be no space before or after the equals sign.

Actions can be used to call certain functions of the simulator. It is currently not possible
to pass parameters to actions.

Note that the simulator will normally exit, after all commands have been processed. To open a GUI
with the current state after all commands have been processed use --gui in the arguments.

Script commands may be specified either directly on the command line using the --do option, or can
be read from a file using --do-file. Both methods support multiple commands and can be used multiple
times. Command files must have one command per line. Examples:

--do r0=0 run_sim
--do r0=0 --do run_sim
--do-file script.txt
--do r0=0 --do-file script1.txt script2.txt --do r0=1 run_sim
"""

commands = dict()

def run_command(cmd: str, state: simstate.SarSimParameterState):
    """Run the specified command, possibly affecting the state."""

    if '=' in cmd: # assignment
        name, value = cmd.split('=', 1)
        try:
            param = next(x for x in state.get_parameters() if x.name == name)
        except StopIteration:
            print(f"Paramer {name} not found, ignoring")
            return

        state.set_value(param, param.type.parse_string(value))
    else: # action
        action, *params = cmd.split()
        try:
            commands[action](*params)
        except KeyError:
            print(f"Command {action} not found, ignoring")

def script_command(func: Callable):
    # register the function
    commands[func.__name__] = func

# Following are the implementations of various commands. Each must be annotated by @script_command to be callable
# from scripting.

@script_command
def print_date():
    print(f"Current Time: {datetime.now()}")
