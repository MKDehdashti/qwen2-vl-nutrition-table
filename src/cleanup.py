import subprocess
import argparse

def clean_cache(deep: bool = False):
    cmd = ["bash", "cleanup.sh"]
    if deep:
        cmd.append("--deep")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="Perform deep cleanup (remove all caches)")
    args = parser.parse_args()

    print(f"🧹 Running cleanup {'(deep)' if args.deep else ''}...")
    clean_cache(deep=args.deep)
    print("✅ Cleanup complete")
