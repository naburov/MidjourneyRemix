# MidjourneyRemix
This is a repo containing small image mixup demo.  
Based on: https://huggingface.co/lambdalabs/sd-image-variations-diffusers.  
Main instruments: hugging face diffusers with pytorch

<table>
  <tr>
    <td><img src="examples/1/landscape.jpeg"  alt="1" width = 360px height = 360px ></td>
    <td><img src="examples/1/style.jpeg" alt="2" width = 360px height = 360px></td>
    <td><img src="examples/1/result-0.5.jpg" alt="2" width = 360px height = 360px></td>
   </tr> 
   <tr>
    <td><img src="examples/4/gigachad.png"  alt="1" width = 360px height = 220px ></td>
    <td><img src="examples/4/shrek-2.jpg" alt="2" width = 360px height = 220px></td>
    <td><img src="examples/4/result-0.9-1.jpg" alt="2" width = 360px height = 360px></td>
   </tr> 
<table>

## Installation
```
git clone https://github.com/naburov/MidjourneyRemix
cd MidjourneyRemix
<create venv if you want>
pip install -r requirements.txt
```
## Usage
To run:  
```
python main.py </path/to/first/image> </path/to/second/image> --second_image_weight <weight> --backend <backend>
python main.py ./examples/1/landscape.jpeg ./examples/1/style.jpeg --second_image_weight 0.9
```
This should create result.jpg in the root of the repo folder.  
Caution: default device (`--backend` argument) is `mps`, change it if you need to `cuda:0` or `cpu`

## How does it work
This is a latent diffusion pipeline. Generation of image is conditioned on embeddings from CLIP. `lambdalabs/sd-image-variations-diffusers`
implements conditioning on image embeddings instead of text embeddings (as in `CompVis/stable-diffusion-v1-5`).  
Image mixing is done via weightening embeddings of base images.   
```
embedding_for_generation = w * embedding_from_the_first_image + (1 - w) * embedding_from_the_second_image
```
It is possible due to the latent space properties like in `word2vec`.
