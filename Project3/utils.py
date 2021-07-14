import matplotlib.pyplot as plt

total_number_of_episodes = []
number_of_actions_per_episode = []
reward_per_episode = []

def set_globals(episode, number_of_actions, reward):
    total_number_of_episodes.append(episode)
    number_of_actions_per_episode.append(number_of_actions)
    reward_per_episode.append(reward)

def plot_rewards():
    plt.plot(total_number_of_episodes, reward_per_episode)
    plt.xlabel('Number of episodes')
    plt.ylabel('Rewards/Score')
    plt.title('Rewards per episode')
    plt.show()

def plot_actions():
    plt.plot(total_number_of_episodes, number_of_actions_per_episode)
    plt.xlabel('Number of episodes')
    plt.ylabel('Number of actions')
    plt.title('Number of actions per episode')
    plt.show()