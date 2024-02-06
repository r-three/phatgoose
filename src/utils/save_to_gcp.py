import subprocess


def save_to_gcp(filepath):
    """

    Args:
        should_save_to_gcp:
        filepath:

    Returns:

    """
    subprocess.call(
        f"gsutil "
        f"-m "
        f"-o GSUtil:parallel_composite_upload_threshold=150M "
        f"cp -r {filepath} gs://co-lm/{filepath}",
        shell=True,
    )
