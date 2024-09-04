from flexibuff import FlexibleBuffer


if __name__ == "__main__":
    import numpy as np

    obs = np.array(
        [  # [agent, timestep, obs]
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ],
            [
                [0.1, 1.1, 2.1],
                [1.1, 2.1, 3.1],
                [2.1, 3.1, 4.1],
            ],
        ]
    )

    fb = FlexibleBuffer(
        num_steps=5,
        obs_size=3,
        action_mask=True,
        discrete_action_cardinalities=[2, 3],
        continious_action_dimension=2,
        path="../test_save/",
        name="all_attributes",
        n_agents=2,
        state_size=3,
        global_reward=True,
        global_auxiliary_reward=True,
        individual_reward=True,
        individual_auxiliary_reward=True,
        log_prob_discrete=True,
        log_prob_continuous=1,
        memory_weights=True,
    )

    # print(fb)
    fb.save_transition(
        obs=np.copy(obs[:, 0, :]),
        obs_=np.copy(obs[:, 1, :]),
        terminated=False,
        discrete_actions=np.array([[0, 1], [1, 2]]),
        continuous_actions=np.array([[0.1, 0.5], [0.2, 1.0]]),
        global_reward=1.0,
        global_auxiliary_reward=0.1,
        individual_rewards=0.5,
        individual_auxiliary_rewards=0.4,
        action_mask=[np.array([[0, 1], [1, 0]]), np.array([[0, 0, 1], [0, 1, 0]])],
        action_mask_=[np.array([[0, 0], [0, 0]]), np.array([[0, 1, 0], [1, 0, 0]])],
        memory_weight=1.5,
        continuous_log_probs=[[-0.1], [-0.2]],
        discrete_log_probs=[[-0.1, -0.1], [-0.2, -0.2]],
        state=np.copy(obs[0, 0, :]),
        state_=np.copy(obs[1, 0, :]),
    )
    fb.save_to_drive()
    fb.load_from_drive()

    print(fb)
