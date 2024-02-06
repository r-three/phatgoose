# Entry point for single process training and evaluation
import json
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Optional

import gin
import pandas as pd  # noqa: F401 # to get it running on the cluster

import src.utils.logging as logging
from src.utils.gin import get_scope_defined_objects, report_scope_defined_objects
from src.utils.set_seeds import set_seeds


@gin.configurable(
    allowlist=[
        "exp_name",
        "procedure_exec_order",
        "recover_from_checkpoint",
        "global_seed",
        "logging_backend",
    ]
)
def main(
    exp_name: str = "debug",
    procedure_exec_order: List[str] = [],
    recover_from_checkpoint: Optional[str] = None,
    global_seed: Optional[int] = None,
    logging_backend=None,
):
    # Logging setup
    exp_name = os.path.expandvars(exp_name)
    logging.logger_setup(backend=logging_backend, exp_name=exp_name)
    logging.print_plate(f"Experiment [{exp_name}] Start")
    print(f"Time: {datetime.now()}")
    print(f"Run [{exp_name}] with the following configuration:")
    print(f"\texp_name: {exp_name}")
    print(f"\tprocedure_exec_order: {procedure_exec_order}")
    print(f"\trecover_from_checkpoint: {recover_from_checkpoint}")
    print(f"\tglobal_seed: {global_seed}")

    # Set global seeds
    if global_seed is not None:
        set_seeds(global_seed)

    # Build and link datasets, models and procedures
    logging.print_plate("Build and Link Objects")
    for procedure_name in procedure_exec_order:
        get_scope_defined_objects(procedure_name)
    logging.print_plate("Object Inventory")
    report_scope_defined_objects()

    # Optionally resume from the checkpoint
    if recover_from_checkpoint:
        raise NotImplementedError("Checkpointing is not implemented yet")
        with open(recover_from_checkpoint, "r") as checkpoing_file:
            checkpoint_info = json.load(checkpoing_file)
        procedure_status = checkpoint_info["procedure_status"]
        object_checkpoint_path_mapping = checkpoint_info[
            "object_checkpoint_path_mapping"
        ]

        for (
            scope_name,
            object_checkpoint_path,
        ) in object_checkpoint_path_mapping.items():
            get_scope_defined_objects(scope_name).recover_states(object_checkpoint_path)
    else:
        procedure_status = ["waiting" for _ in procedure_exec_order]

    # Run procedures
    logging.print_plate("Run Procedures")
    for procedure_idx, procedure_name in enumerate(procedure_exec_order):
        if procedure_status[procedure_idx] == "finished":
            continue
        procedure_status[procedure_idx] = "running"
        get_scope_defined_objects(procedure_name).run()
        procedure_status[procedure_idx] = "finished"

    # Exit
    logging.print_plate(f"Experiment [{exp_name}] Finished")
    print(f"Time: {datetime.now()}")
    logging.logger_close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gin_files", nargs="+", required=True)
    parser.add_argument("--gin_bindings", nargs="+", default=[])
    args = parser.parse_args()

    for file_name in args.gin_files:
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} not found")
    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)
    main()
