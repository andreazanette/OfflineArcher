import torch 

def batch_simulate_on_environment(policy, env, verbose = True):
    if verbose:
        print("*** In batch_simulate_on_environment ***")
    
    from Dataset import Trajectory, TrajectoryDataset
    from math import ceil

    dataset = TrajectoryDataset()

    trajectories = [Trajectory() for _ in range(env.bsize)]
    batch_obs = env.reset() 
    batch_done = [False,]*env.bsize
    while not all(batch_done):
        with torch.no_grad():
            actions = policy(batch_obs)
        batch_feedback = env.step(actions)
        for i, feedback in zip(range(env.bsize), batch_feedback):
            if feedback is None:
                continue

            next_obs, r, done = feedback
            
            trajectories[i].append({"observation": batch_obs[i],
                                    "action": actions[i], 
                                    "reward": r,
                                    "next_observation": next_obs, 
                                    "done": done, 
                                    })
            batch_obs[i] = next_obs
            batch_done[i] = done
    for trajectory in trajectories:
        dataset.append_trajectory(trajectory)
        print(trajectory.transitions[-1].next_observation)

    dataset.check_consistency()
    if verbose:
        print("Data Coollection is Complete. Returns: \n", dataset.get_all_trajectory_returns(), "\n with mean: ",dataset.mean_trajectory_return(), "\n" )
    return dataset
