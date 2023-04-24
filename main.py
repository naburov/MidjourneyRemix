import argparse

from pipeline import ImageMixupPipeline
from PIL import Image
from torchvision.transforms import transforms


def mix(image1_name, image2_name, device, weight):
    device = device
    sd_pipe = ImageMixupPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    sd_pipe = sd_pipe.to(device)

    im = Image.open(image1_name)
    im2 = Image.open(image2_name)
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

    out = sd_pipe(inp, inp1, weight)
    out["images"][0].save("result.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("image1_name", help="path to the first image", type=str)
    parser.add_argument("image2_name", help="path to the second image", type=str)
    parser.add_argument("--second_image_weight", help="float number in [0..1]", type=float, default=0.5)
    parser.add_argument("--backend", help="torch backend", type=str, default='mps')

    args = parser.parse_args()
    print(args)
    mix(args.image1_name, args.image2_name, args.backend, args.second_image_weight)
