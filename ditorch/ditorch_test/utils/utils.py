import subprocess
import os

def sparse_checkout(repo_url, destination, paths, branch="main", depth=1):
    destination = os.path.abspath(
        destination
    )  # Ensure the destination is an absolute path

    try:
        # Perform a shallow clone with the specified depth and branch
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--depth",
                str(depth),
                "--single-branch",
                "-b",
                branch,
                repo_url,
                destination,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git clone failed: {e}")

    git_dir = os.path.join(destination, ".git")

    try:
        # Initialize sparse checkout and set the specified paths
        subprocess.run(
            [
                "git",
                "--git-dir",
                git_dir,
                "--work-tree",
                destination,
                "sparse-checkout",
                "init",
                "--cone",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "--git-dir",
                git_dir,
                "--work-tree",
                destination,
                "sparse-checkout",
                "set",
            ]
            + paths,
            check=True,
        )
        # Checkout the files in the sparse checkout
        subprocess.run(
            ["git", "--git-dir", git_dir, "--work-tree", destination, "checkout"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Sparse checkout failed: {e}")

    print("Sparse checkout 完成!")


