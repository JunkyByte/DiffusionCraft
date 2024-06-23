import os
from PIL import Image
import glob

def create_image_grid(image_paths, output_path):
    images = [Image.open(image_path) for image_path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    assert [w == widths[0] for w in widths]
    total_width = max(widths)
    total_height = sum(heights)

    grid_image = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for im in images:
        grid_image.paste(im, (0, y_offset))
        y_offset += im.size[1]

    grid_image.save(output_path)

# Horizontal
def create_image_grid(image_paths, output_path):
    images = [Image.open(image_path) for image_path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    assert [h == heights[0] for h in heights], "All images must have the same height"
    total_width = sum(widths)
    total_height = max(heights)

    grid_image = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    for im in images:
        grid_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    grid_image.save(output_path)


def create_grids(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index in range(11):
        files = [[] for _ in range(8)]
        for reps in range(0, 8):
            for strength_dir in sorted(glob.glob(os.path.join(base_dir, 'variation_strength_*')), key=lambda x: float(x[x.rfind('_') + 1:])):
                f = glob.glob(os.path.join(strength_dir, f'index_{index}', 'samples/', f'*_{reps}.png'))
                files[reps].append(f[0])
        
                out = os.path.join(output_dir, f'seed_{index}')
                if not os.path.exists(out):
                    os.makedirs(out)

            grid_image_output = os.path.join(out, f"{reps}_grid.png")
            create_image_grid(files[reps], grid_image_output)

if __name__ == "__main__":
    base_dir = 'output/variation_strength/'
    output_dir = 'output/variation_strength/results/'
    create_grids(base_dir, output_dir)