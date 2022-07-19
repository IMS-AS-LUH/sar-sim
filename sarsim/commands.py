# This module contains simple commands that may be called from scripts (and are used by the GUI)

from typing import Callable, Optional

import argparse
from dataclasses import dataclass
from datetime import datetime

from sarsim import sardata, simjob, simscene, simstate, profiling, simstate

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
--do "load_capture cap.sardata"
--do-file script.txt
--do r0=0 --do-file script1.txt script2.txt --do r0=1 run_sim
"""

@dataclass
class ProgramState:
    """This class serves as a container for the (global) state of the program."""
    args: argparse.Namespace # command line args
    simstate: simstate.SarSimParameterState
    scene: simscene.SimulationScene
    loaded_dataset: Optional[sardata.SarData] = None
    sim_result: Optional[simjob.SimResult] = None
    color_preset: str = 'jet'

commands = dict()

def run_command(cmd: str, pstate: ProgramState):
    """Run the specified command, possibly affecting the state."""

    if '=' in cmd: # assignment
        name, value = cmd.split('=', 1)
        try:
            param = next(x for x in pstate.simstate.get_parameters() if x.name == name)
        except StopIteration:
            print(f"Paramer {name} not found, ignoring")
            return

        pstate.simstate.set_value(param, param.type.parse_string(value)) # type: ignore
    else: # action
        action, *params = cmd.split()
        try:
            commands[action](pstate, *params) # call action with parameters, first is always the pstate
        except KeyError:
            print(f"Command {action} not found, ignoring")

def script_command(func: Callable):
    """Decorator for functions that can be called from a script"""
    # register the function
    commands[func.__name__] = func
    return func

# Following are the implementations of various commands. Each must be annotated by @script_command to be callable
# from scripting.

@script_command
def print_date(pstate: ProgramState):
    print(f"Current Time: {datetime.now()}")

@script_command
def load_capture(pstate: ProgramState, path: str):
    sd = sardata.SarData.import_from_directory(path)
    pstate.simstate = sd.sim_state
    pstate.loaded_dataset = sd

    print(f'Loaded SarData from: {path}')
    if not sd.has_range_compressed_data:
        assert sd.fmcw_lines is not None
        print(f'Raw FMCW: {len(sd.fmcw_lines)} lines of {len(sd.fmcw_lines[0])} samples')
    else:
        assert sd.rg_comp_data is not None
        print(f'Range compressed: {len(sd.rg_comp_data)} lines of {len(sd.rg_comp_data[0])} complex samples')

@script_command
def unload_capture(pstate: ProgramState):
    pstate.loaded_dataset = None
    pstate.simstate = simstate.create_state()

@script_command
def set_color_preset(pstate: ProgramState, color_preset: str):
    pstate.color_preset = color_preset

@script_command
def load_param_file(pstate: ProgramState, filename: str):
    pstate.simstate = simstate.SarSimParameterState.read_from_file(filename)

@script_command
def save_param_file(pstate: ProgramState, filename: str):
    pstate.simstate.write_to_file(filename)

@script_command
def run_sim(pstate: ProgramState):
    ts = profiling.TimeStamper()
    pstate.sim_result = simjob.run_sim(pstate.simstate, pstate.scene, ts, None, pstate.loaded_dataset, pstate.args.gpu)

