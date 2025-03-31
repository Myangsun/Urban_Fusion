"""
UrbanFusion Prototype - Environment Setup
This file sets up the environment for the UrbanFusion system
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories
def setup_environment():
    """Set up the environment for the UrbanFusion system"""
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/sample", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Other configuration\n")
            f.write("DEBUG=True\n")
    
    print("Environment setup complete.")

if __name__ == "__main__":
    setup_environment()
