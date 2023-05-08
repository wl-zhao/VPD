import argparse
import copy
import os
import os.path as osp
import time
import torch
# import denseclip
from  tqdm import tqdm

from glob import glob
import json

def main():
    import sys
    sys.path.append('../')
    from vpd.models import FrozenCLIPEmbedder

    paths = glob('nyu_depth_v2/official_splits/train/*')

    class_name = [path.split('/')[-1] for path in paths]

    print(class_name)

    with open('nyu_class_list.json', 'w') as f:
        f.write(json.dumps(class_name))
    
    imagenet_classes = [name.replace('_', ' ') for name in class_name]

    # imagenet_classes = ['close',  'far', 'nearby', 'in middle distance', 'far away', 'in background']

    # mid = ['object ', 'something ', 'stuff ', 'guy ', 'people ']

    imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    ]

    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    text_encoder = FrozenCLIPEmbedder(max_length=20)
    text_encoder.cuda()

    classnames = imagenet_classes


    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            texts = texts + [template.format(classname) for template in imagenet_templates] #format with class
            print(texts[0])
            class_embeddings = text_encoder.encode(texts).detach().mean(dim=0)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)

    print(zeroshot_weights.shape)
    torch.save(zeroshot_weights.cpu(), 'nyu_class_embeddings.pth')

if __name__ == '__main__':
    main()
