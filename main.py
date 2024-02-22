import narla
import torch
import numpy as np

# Parse command line args into narla.settings
narla.parse_args()

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create the Environment
environment = narla.environments.GymEnvironment(
    name=narla.settings.environment,
    render=narla.settings.render
)
observation = environment.reset()


# Build the MultiAgentNetwork based on settings
network = narla.multi_agent_network.MultiAgentNetwork(
    observation_size=environment.observation_size,
    learning_rate=narla.settings.learning_rate,
    number_of_actions=environment.action_space.number_of_actions,
    number_of_layers=narla.settings.number_of_layers,
    number_of_neurons_per_layer=narla.settings.number_of_neurons_per_layer
)


for episode_number in range(1, narla.settings.maximum_episodes + 1):

    observation = environment.reset()
    for count in narla.count():
        # The network computes an action based on the observation
        action = network.act(observation)

        # Execute the action in the environment
        observation, reward, terminated = environment.step(action)

        # Distribute reward information to all layers
        network.distribute_to_layers(**{
            narla.history.reward_types.TASK_REWARD: reward,
            narla.history.saved_data.TERMINATED: terminated
        })

        if terminated:
            print("Episode:", episode_number, "total reward:", environment.episode_reward, flush=True)

            # Record the reward history for the episode
            network.record(**{
                narla.history.reward_types.EPISODE_REWARD: environment.episode_reward,
                narla.history.saved_data.EPISODE_COUNT: episode_number
            })
            break

    # Network learns based on episode
    network.learn()

    if episode_number % narla.settings.save_every:
        narla.io.save_history_as_data_frame(
            name="results",
            history=network.history
        )


narla.io.save_history_as_data_frame(
    name="results",
    history=network.history
)
print("done", flush=True)
