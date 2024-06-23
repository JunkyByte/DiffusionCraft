import torch
import h5py
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from PIL import Image
from scipy.ndimage import zoom
import numpy as np

import sys
sys.path.append('./src/imagebind')

sensor_to_params = {
    "kv1": {
        "baseline": 0.075,
    },
    "kv1_b": {
        "baseline": 0.075,
    },
    "kv2": {
        "baseline": 0.075,
    },
    "realsense": {
        "baseline": 0.095,
    },
    "xtion": {
        "baseline": 0.095, # guessed based on length of 18cm for ASUS xtion v1
    },
}


def convert_depth_to_disparity(depth, focal_length, sensor_type, min_depth=0.01, max_depth=50):
    baseline = sensor_to_params[sensor_type]["baseline"]
    # depth_in_meters = depth / 1000.
    depth_in_meters = depth
    if min_depth is not None:
        depth_in_meters = depth_in_meters.clip(min=min_depth, max=max_depth)
    disparity = baseline * focal_length / depth_in_meters
    return torch.from_numpy(disparity).float()


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_list = ["A very beautiful canine.", "A car", "A bird", "A fat cartoon rabbit leaving his home", "a fat animal with big ears is gray", 'intern of a room']
    image_paths = ["samples/imgs/dog_image.jpg", "samples/imgs/car_image.jpg", "samples/imgs/bird_image.jpg", 'samples/imgs/test_depth.png']
    audio_paths = ["samples/audio/dog_audio.wav", "samples/audio/car_audio.wav", "samples/audio/bird_audio.wav"]
    video_paths = ["samples/imgs/rabbit_cartoon.mp4"]

    depth_files = ['./samples/depth/01441.h5']
    with h5py.File(depth_files[0], 'r') as f:
        depth = np.array(f['depth'])
        depth = zoom(depth, (224/depth.shape[0], 224/depth.shape[1]), order=1)
        # rgb = Image.fromarray(np.array(f['rgb']).transpose(1, 2, 0), mode='RGB')
        # rgb.save('./samples/imgs/test_depth.png')
        disparity = convert_depth_to_disparity(depth, 518.85790117450188, 'kv1', min_depth=0.01, max_depth=50).unsqueeze_(dim=0).to(device)

    # Instantiate model
    model = imagebind_model.imagebind_huge(ckpt_path='/home/adryw/dataset/imagecraft/imagebind_huge.pth', pretrained=True)
    model.eval()
    model.to(device)

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
        ModalityType.DEPTH: disparity[None],
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Audio x Text: ",
        torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Vision x Audio: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
    )
    print(
        "Video x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Depth x Vision: ",
        torch.softmax(embeddings[ModalityType.DEPTH] @ embeddings[ModalityType.VISION].T, dim=-1),
    )
    print(
        "Depth x Text: ",
        torch.softmax(embeddings[ModalityType.DEPTH] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )

    # Test with video
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(
        "Vision x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )