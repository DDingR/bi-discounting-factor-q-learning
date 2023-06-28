import matplotlib
import matplotlib.pyplot as plt

def plot_rewards(reward_list, show_result=False):
    plt.figure(1)
    reward_list = torch.tensor(reward_list, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_list) >= 100:
        means = reward_list.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())