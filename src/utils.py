# %%
import matplotlib.pyplot as plt

# %%
def print_sample(sample, generated_img, row=0):
    input = sample['x']
    ground_truth = sample['y']
    generated_img = generated_img
    imgs = [input, ground_truth, generated_img]
    _, axs = plt.subplots(ncols=3, squeeze=False)
    axs[0, 0].set_title('input')
    axs[0, 1].set_title('ground_truth')
    axs[0, 2].set_title('output')
    for i, img in enumerate(imgs):
        axs[row, i].imshow(img.permute(1, 2, 0))
        axs[row, i].set_xticklabels([])
        axs[row, i].set_yticklabels([])
    plt.show()
