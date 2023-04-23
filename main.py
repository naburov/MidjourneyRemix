from pipeline import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision.transforms import transforms

device = "mps"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

im = Image.open("/Users/burovnikita/PycharmProjects/MidjourneyRemix/images/landscape.jpeg")
im2 = Image.open("/Users/burovnikita/PycharmProjects/MidjourneyRemix/images/style.jpeg")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)
inp1 = tform(im2).to(device).unsqueeze(0)

out = sd_pipe(inp, inp1, guidance_scale=2)
out["images"][0].save("result.jpg")