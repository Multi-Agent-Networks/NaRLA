import narla


# Parse command line args into narla.settings
narla.parse_args()


# Create the Environment
environment = narla.environments.GymEnvironment(
    name=narla.settings.environment,
    render=narla.settings.render
)
observation = environment.reset()


# Build the MultiAgentNetwork based on settings
network = narla.multi_agent_network.MultiAgentNetwork(
    observation_size=environment.observation_size,
    number_of_actions=environment.action_space.number_of_actions,
    number_of_layers=narla.settings.number_of_layers,
    number_of_neurons_per_layer=narla.settings.number_of_neurons_per_layer
)

for episode_number in range(narla.settings.maximum_episodes):

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


narla.io.save_history_as_data_frame(
    name="network",
    history=network.history
)
print("done", flush=True)
