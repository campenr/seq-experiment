
def plot_seq_experiment(sxp, *args, **kwargs):

    # set custom plotting parameters if user has not specified them
    default_args = {
        'kind': 'bar',
        'width': 0.8,
        'linewidth': 1,
        'edgecolor': 'black',
    }
    for arg, value in default_args.items():
        if arg not in kwargs.keys():
            kwargs[arg] = value

    ax = sxp.features.transpose().plot(*args, **kwargs)

    # tidy legend
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    return ax
