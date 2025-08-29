from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN_TOURISM"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="CRR79/TourismPackage-Purchase-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
