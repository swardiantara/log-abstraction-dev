import os
from huggingface_hub import Repository
from sentence_transformers import SentenceTransformer

# Path to your trained model
model_path = os.path.join('experiments', 'embeddings')

# Load the trained model
model = SentenceTransformer(model_path)

# Save the model to the model_path (ensure it includes all necessary files)
model.save(model_path)

# Specify your Hugging Face model repository name
repo_name = "swardiantara/drone-sbert"

# Create a repository
repo = Repository(local_dir=model_path, clone_from=repo_name)

# Push the model to the Hugging Face Hub
repo.push_to_hub(commit_message="Initial commit of drone log embedding model")
