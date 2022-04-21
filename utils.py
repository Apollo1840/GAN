import matplotlib.pyplot as plt


# create a plot of generated images (reversed grayscale)
def show_plot(examples, n, with_channel=True):
    """

    :param: number of rows and columns
    """
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        if with_channel:
            # shape = (n_sample, x_axis, y_axis, channel)
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        else:
            # shape = (n_sample, x_axis, y_axis)
            plt.imshow(examples[i], cmap='gray_r')

    plt.show()
