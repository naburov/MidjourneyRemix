from diffusers import StableDiffusionPipeline
from PIL import Image

image1 = r'/Users/burovnikita/PycharmProjects/MidjourneyRemix/images/kowalski.jpeg'

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
encoding = pipe.tokenizer(prompt, return_tensors='pt').to("mps")

# image = Image.open(image1)
# First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
_ = pipe(prompt, num_inference_steps=1)
encoded = pipe.text_encoder(**encoding)
# Results match those from the CPU device after the warmup pass.
print(encoded['last_hidden_state'].shape)
generated = pipe(prompt_embeds=encoded['last_hidden_state']).images[0]
generated.save("astronaut_rides_horse.png")
