#!/usr/bin/env python
# coding: utf-8

# # Use the CartPole-v0 environment and write a program to :
# 
# 

# # a. Implement the CartPole environment for a certain number of steps

# In[2]:


import gym

env = gym.make('CartPole-v0')
num_steps = 1000

observation = env.reset()
reward_sum = 0

for i in range(num_steps):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    reward_sum += reward

    if done:
        observation = env.reset()
        print(f"Episode finished after {i+1} steps with total reward {reward_sum}")
        reward_sum = 0

env.close()


# 

# In[10]:


import gym
import numpy as np

def play_cartpole(steps):
    env = gym.make('CartPole-v0', render_mode="rgb_array")
    max_steps = steps
    state = env.reset()
    episode = 0
    while True:
        env.render()
        action = env.action_space.sample()
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        print("Episode: {} - Step: {} - Reward: {} - Done: {}".format(episode, max_steps, reward, done))
        if done or max_steps == 0:
            state = env.reset()
            episode += 1
            print("Episode {} finished.".format(episode))
        max_steps -= 1
        if max_steps == 0:
            break
    env.close()
play_cartpole(100)


# # b. Implement the CartPole environment for a certain number of episodes

# In[4]:


import gym

env = gym.make('CartPole-v0')
num_episodes = 100

for episode in range(num_episodes):
    observation = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        observation, reward, done, _, _ = env.step(action)
        episode_reward += reward

    print(f"Episode {episode + 1} finished with total reward {episode_reward}")

env.close()


# In[ ]:





# In[12]:


import gym
import numpy as np

def play_cartpole(episodes):
    env = gym.make('CartPole-v0')
    max_episodes = episodes
    state = env.reset()
    step = 0
    for episode in range(max_episodes):
        env.render()
        action = env.action_space.sample()
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        print("Episode: {} - Step: {} - Reward: {} - Done: {}".format(episode, step, reward, done))
        if done:
            state = env.reset()
            print("Episode {} finished.".format(episode))
        step += 1
    env.close()
play_cartpole(10)


# # c. Compare and comment on the rewards earned for both approaches.

# The rewards earned in both approaches can vary significantly. This is because in approach a, the agent takes a random action in each step, resulting in unpredictable and varying rewards. On the other hand, in approach b, the agent resets the environment at the end of each episode, ensuring that each episode is independent of the previous ones. However, this approach does not provide the agent with an opportunity to learn from its mistakes, which may be beneficial in the long run.
# 
# The results of both approaches will also depend on the number of steps and episodes chosen for the experiment. If a smaller number of steps or episodes is chosen, the cumulative reward may be lower, as the agent does not have enough time to learn from the environment. However, if a larger number of steps or episodes is chosen, the cumulative reward may be higher, as the agent has more opportunities to learn from its mistakes and make better decisions.
# 
# Ultimately, the choice between the two approaches will depend on the specific requirements of your experiment and the desired level of performance from your agent.

# # d. Plot the cumulative reward of the games and note down the results.
# 

# In[6]:


import matplotlib.pyplot as plt

rewards_per_episode = []

for episode in range(num_episodes):
    done = False
    obs = env.reset()
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ , _= env.step(action)
        episode_reward += reward

    rewards_per_episode.append(episode_reward)

plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward per Episode')
plt.show()


# In[ ]:




