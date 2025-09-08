import subprocess

def clean_cache(deep: bool = False):
    cmd = ["bash", "/workspace/projects/nutrition-table/cleanup.sh"]
    if deep:
        cmd.append("--deep")
    subprocess.run(cmd, check=True)
