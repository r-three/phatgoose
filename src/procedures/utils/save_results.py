import os

import gin


@gin.configurable
def save_results(results, save_dir, step=None, overwrite=False):
    save_dir = os.path.expandvars(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if step is not None:
        head_line = f"Training step: {step}\n"
        save_path = save_dir + f"/results_in_training.txt"
    else:
        head_line = "Final results:\n"
        save_path = save_dir + "/results.txt"
    if overwrite:
        open_mode = "w+"
    else:
        open_mode = "a+"
    with open(save_path, open_mode) as f:
        f.write(head_line)
        f.write(str(results))
        f.write("\n\n")
