# Prediction interface for Cog ⚙️
# https://cog.run/python

import subprocess
import time
from cog import BasePredictor, Input, Path
import os
import torch
from diffusers import StableDiffusionPipeline

MODEL_CACHE = "model_cache"
BASE_URL=  f"https://weights.replicate.delivery/default/test-sd-15/{MODEL_CACHE}/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
            
        # Set environment variables for model caching
        os.environ["HF_HOME"] = MODEL_CACHE
        os.environ["TORCH_HOME"] = MODEL_CACHE
        os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
        os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
            
        model_files = ["models--sd-legacy--stable-diffusion-v1-5.tar"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
                
        # Load the model
        model_id = "sd-legacy/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            cache_dir=MODEL_CACHE
        )
        self.pipe = self.pipe.to("cuda")
                
    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale for text conditioning", ge=1.0, le=20.0, default=7.5
        ),
    ) -> Path:
        """Generate an image from a text prompt"""
        # Generate image
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Save the output image
        output_path = Path("output.png")
        image.save(output_path)
        
        return output_path