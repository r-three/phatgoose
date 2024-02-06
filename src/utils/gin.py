import os

import gin

import src.utils.logging as logging

scope_defined_objects_dict = {
    "__data__": [],
    "__models__": [],
    "__procedures__": [],
}


def get_scope_defined_objects(scope_name):
    scope_name = os.path.expandvars(scope_name)
    assert scope_name[:2] in [
        "D/",
        "M/",
        "P/",
    ], f"Please use a scope name starting with D/, M/, or P/ for data, model, and procedure, respectively. Got {scope_name}."
    if scope_name not in scope_defined_objects_dict:
        with gin.config_scope(scope_name):
            logging.print_single_bar()
            print(f"Building {scope_name}...")
            scope_defined_objects_dict[scope_name] = build(scope_name=scope_name)
            print(f"Done building {scope_name}.")
        if scope_name.startswith("D/"):
            scope_defined_objects_dict["__data__"].append(scope_name)
        elif scope_name.startswith("M/"):
            scope_defined_objects_dict["__models__"].append(scope_name)
        elif scope_name.startswith("P/"):
            scope_defined_objects_dict["__procedures__"].append(scope_name)
            print(f"Linking {scope_name} to find other top-level objects...")
            scope_defined_objects_dict[scope_name].link()
            with gin.config_scope(scope_name):
                print(f"Done linking {scope_name}, run late_init...")
                scope_defined_objects_dict[scope_name].late_init()

    return scope_defined_objects_dict[scope_name]


def report_scope_defined_objects():
    print("Listing all defined objects...")
    for type in ["__data__", "__models__", "__procedures__"]:
        print(f"Defined {type.strip('_')}")
        count = len(scope_defined_objects_dict[type])
        for idx, scope_name in enumerate(scope_defined_objects_dict[type]):
            print(f"\t({idx+1}/{count}) {scope_name}")
            for line in scope_defined_objects_dict[scope_name].get_description():
                print(f"\t\t{line}")


@gin.configurable(allowlist=["cls"])
def build(cls, scope_name):
    return cls(name=scope_name)


def share_memory():
    all_objects = (
        scope_defined_objects_dict["__data__"]
        + scope_defined_objects_dict["__models__"]
        + scope_defined_objects_dict["__procedures__"]
    )
    for scope_name in all_objects:
        scope_defined_objects_dict[scope_name].share_memory()
