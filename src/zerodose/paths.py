"""This file contains paths to the parameter files.\
    Please refer to the readme on where to get the parameters.\
        Save them in this folder: ~/.zerodose_params."""
import os


# please refer to the readme on where to get the parameters. Save them in this folder:
folder_with_parameter_files = os.path.join(
    os.path.expanduser("~"), ".zerodose_data", "model_params"
)
folder_with_templates = os.path.join(
    os.path.expanduser("~"), ".zerodose_data", "templates"
)
