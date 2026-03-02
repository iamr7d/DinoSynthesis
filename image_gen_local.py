"""
Local image generation using SDXL-Turbo (stabilityai/sdxl-turbo).
- 1-4 step generation, very fast on GPU (~1s on RTX 5090)
- Lazy-loaded: model downloads/loads on first request
- Returns base64-encoded PNG
"""
import io
import base64
import threading
import torch
from PIL import Image

MODEL_ID    = "stabilityai/sdxl-turbo"
_pipe       = None
_load_lock  = threading.Lock()
_infer_lock = threading.Lock()  # pipeline is not thread-safe


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    with _load_lock:
        if _pipe is not None:
            return _pipe
        print("[image_gen] Loading SDXL-Turbo …")
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            variant="fp16",
        )
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        _pipe = pipe
        print("[image_gen] SDXL-Turbo ready ✓")
    return _pipe


# Prompt templates per animal group ─────────────────────────────────────────
_PROMPTS = {
    "Bird": [
        "photorealistic feathered Archaeopteryx dinosaur perched on Mesozoic fern, iridescent blue-green plumage, golden hour light, BBC wildlife photography, hyperdetailed, 8k",
        "ancient Microraptor gliding between Cretaceous trees, vivid iridescent feathers, dappled jungle light, National Geographic quality",
        "Cretaceous bird-like theropod mid-flight, detailed feathers, dramatic cloudy sky, ultra realistic nature documentary still",
    ],
    "Reptile": [
        "massive prehistoric Sarcosuchus crocodilian emerging from Mesozoic swamp, detailed scales, rippling water, dramatic low angle, photorealistic 8k",
        "giant Deinosuchus resting on Cretaceous riverbank, mossy trees, atmospheric mist, BBC Planet Earth cinematography, ultra detailed",
        "prehistoric alligator-like reptile in primordial jungle, golden sunset, National Geographic quality, hyperrealistic scales",
    ],
    "Mass": [
        "diverse dinosaurs fleeing Chicxulub asteroid impact, fire and ash sky, epic cinematic wide shot, photorealistic 8k, Jurassic Park quality",
        "prehistoric landscape at mass extinction, dinosaur silhouettes against burning horizon, asteroid glow, ultra detailed",
        "Cretaceous-Paleogene boundary scene, sky on fire, last dinosaurs, dramatic shadows, award-winning digital art",
    ],
    "Blend": [
        "hybrid transitional creature blending bird and crocodilian features, seamless feather-scale fusion, Mesozoic jungle, hyperrealistic scientific illustration",
        "speculative palaeontology: chimeric dinosaur bridging avian and reptilian lineage, detailed anatomy, museum diorama lighting",
        "evolutionary transition: feathered reptile dinosaur, vivid plumage and scales, Cretaceous rainforest, 8k render",
    ],
}

NEGATIVE_PROMPT = (
    "text, watermark, logo, blurry, cartoon, anime, painting, drawing, "
    "low quality, bad anatomy, deformed, ugly, duplicate, human, person"
)


def build_prompt(group: str, x: float, y: float, z: float) -> str:
    pool = _PROMPTS.get(group, _PROMPTS["Blend"])
    dist = (x**2 + y**2 + z**2) ** 0.5
    idx  = int(dist * 3) % len(pool)
    base = pool[idx]
    size_tag = "enormous apex predator, " if dist > 3.5 else ("small nimble creature, " if dist < 1.5 else "")
    mood_tag = "vibrant living ecosystem, " if z > 0 else "ancient fossil-like, "
    return size_tag + mood_tag + base


def generate(prompt: str, seed: int = 42, steps: int = 4,
             width: int = 512, height: int = 512) -> str:
    """
    Generate an image and return as base64-encoded PNG string.
    Steps=1 is fastest (SDXL-Turbo is designed for 1-4 steps).
    guidance_scale must be 0.0 for turbo.
    """
    pipe = _load_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(seed % (2**32))
    with _infer_lock:  # pipeline is not thread-safe
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=0.0,
                width=width,
                height=height,
                generator=generator,
            )
    img: Image.Image = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


if __name__ == "__main__":
    # Quick smoke-test
    import sys
    grp   = sys.argv[1] if len(sys.argv) > 1 else "Bird"
    prompt = build_prompt(grp, 1.0, 0.5, 0.2)
    print("Prompt:", prompt)
    b64 = generate(prompt, seed=42, steps=4)
    with open("test_output.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("Saved test_output.png")
