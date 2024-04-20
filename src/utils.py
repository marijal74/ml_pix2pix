# %%
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# %%
def print_sample(sample):
    x = sample['x'].permute(1, 2, 0)
    y = sample['y'].permute(1, 2, 0)
    grid = make_grid(x, y)
    plt.imshow(grid)
    plt.show()

