import narla

# Parse command line args into narla.experimental_settings
settings = narla.settings.parse_args()


# Create the Environment
environment = narla.environments.GymEnvironment(name=settings.environment_settings.environment, render=settings.environment_settings.render)
observation = environment.reset()


# Build the MultiAgentNetwork based on settings
network = narla.multi_agent_network.MultiAgentNetwork(
    observation_size=environment.observation_size,
    number_of_actions=environment.action_space.number_of_actions,
    network_settings=settings.multi_agent_network_settings,
)


for episode_number in range(1, settings.trial_settings.maximum_episodes + 1):

    observation = environment.reset()
    for count in narla.count():
        # The network computes an action based on the observation
        action = network.act(observation)

        # Execute the action in the environment
        observation, reward, terminated = environment.step(action)

        # Distribute reward information to all layers
        network.distribute_to_layers(
            **{
                narla.rewards.RewardTypes.TASK_REWARD: reward,
                narla.history.saved_data.TERMINATED: terminated,
            }
        )

        if terminated:
            print("Episode:", episode_number, "total reward:", environment.episode_reward, flush=True)

            # Record the reward history for the episode
            network.record(
                **{
                    narla.history.saved_data.EPISODE_REWARD: environment.episode_reward,
                    narla.history.saved_data.EPISODE_COUNT: episode_number,
                }
            )
            break

    # Network learns based on episode
    network.learn()

    if episode_number % settings.trial_settings.save_every:
        narla.io.save_history_as_data_frame(name="results", history=network.history)


narla.io.save_history_as_data_frame(name="results", history=network.history)
print("done", flush=True)
