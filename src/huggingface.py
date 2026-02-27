from pathlib import Path
import torch
import shutil
import os
from huggingface_hub import HfApi, list_repo_files, create_commit, CommitOperationAdd, hf_hub_download

def login():
        from huggingface_hub import login
        import getpass

        token_file = Path('huggingface/.token')
        token_file.parent.mkdir(exist_ok=True, parents=True)

        if token_file.exists():
            with open(token_file, 'rb') as f:
                token = f.read().strip()
            try:
                login(token.decode())
                return
            except Exception as e:
                print(f"Stored HuggingFace token is invalid: {e}")
        token = getpass.getpass("Enter your HuggingFace token: ")
        login(token)
        with open(token_file, 'wb') as f:
            f.write(token.encode())

def publish_model(run_id:str, publish_name:str, repo_id:str = "SchulzR97/FasterRCNN"):
    login()

    repo_files = list_repo_files(repo_id, repo_type='model')
        
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)  # 'exist_ok=True' avoids errors if repo exists
    print("Repo URL:", repo_url)

    cache_dir = Path("huggingface/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(f"runs/{run_id}/checkpoint.ckpt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the run_id ({run_id}) is correct and the model has been trained.")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']

    if f"{publish_name}.pth" in repo_files:
        print(f"Model '{publish_name}.pth' already exists in the repository. It will be overwritten.")
        # api.delete_file(path_in_repo=f"{publish_name}.pth", repo_id=repo_id, repo_type='model')

    model_file = cache_dir / f"{publish_name}.pth"
    torch.save(model_state_dict, model_file)

    commit_add = CommitOperationAdd(path_in_repo=f"{publish_name}.pth", path_or_fileobj=model_file)
    create_commit(repo_id=repo_id, operations=[commit_add], repo_type='model', commit_message=f"Add model statedict for {publish_name}")

def load_state_dict(publish_name:str, repo_id:str = "SchulzR97/FasterRCNN", force_download:bool = False, clear_cache:bool = False):
    #login()

    repo_files = list_repo_files(repo_id, repo_type='model')
    if f"{publish_name}.pth" not in repo_files:
        raise FileNotFoundError(f"Model '{publish_name}.pth' not found in the repository '{repo_id}'. Available files: {repo_files}")
    
    # download model file to cache
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=f"{publish_name}.pth",
        repo_type='model',
        local_dir_use_symlinks=False,
        force_download=force_download
    )
    state_dict = torch.load(model_file, map_location='cpu')
    if clear_cache:
        os.remove(model_file)
    
    return state_dict