import subprocess
import os

def clone_and_checkout(repo_url, commit_hash, target_dir="cloned_repo"):
    """
    Clones a GitHub repository into an existing directory (if applicable) and checks out a specific commit hash.

    Parameters:
        repo_url (str): The GitHub repository URL.
        commit_hash (str): The hash of the commit to check out.
        target_dir (str): The directory to clone the repository into or use if it exists.
    """
    try:
        if os.path.exists(target_dir):
            print(f"Directory {target_dir} already exists. Using it for cloning.")
            os.chdir(target_dir)
            
            # Initialize a new Git repository if not already initialized
            if not os.path.exists(os.path.join(target_dir, ".git")):
                print("Initializing Git repository in the existing directory...")
                subprocess.run(["git", "init"], check=True)
                subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
            
            # Fetch the repository and the specific commit
            print(f"Fetching from {repo_url}...")
            subprocess.run(["git", "fetch", "origin"], check=True)
        else:
            # Clone the repository into the target directory
            print(f"Cloning repository from {repo_url} into {target_dir}...")
            subprocess.run(["git", "clone", "--no-checkout", repo_url, target_dir], check=True)
            os.chdir(target_dir)
        
        # Checkout the specific commit
        print(f"Checking out commit {commit_hash}...")
        subprocess.run(["git", "checkout", commit_hash], check=True)
        print("Checkout complete.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during the git operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    repo_url = "https://github.com/nordwind80/BT-Tracker"  # Replace with your GitHub link
    commit_hash = "558c15b399871c1ca11d0c4ae1eb598e3060931e"  # Replace with your commit hash
    project_name = repo_url.split("/")[-1]

    clone_and_checkout(repo_url, commit_hash, project_name)
