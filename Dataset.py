from torch.utils.data import Dataset
import numpy as np
import copy
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class Transition:
    def __init__(self, observation, action, reward, next_observation, done, **kwargs):
        self.observation       = observation
        self.action            = action
        self.reward            = np.single(reward)
        self.next_observation  = next_observation

        if isinstance(done, bool):
            self.done          = done
        elif done == 'False':
            self.done          = False
        elif done == 'True':
            self.done          = True
        else: 
            raise ValueError
        
        # internal, to see how many times a certain transition was sampled
        self.times_was_sampled = 0
    def as_dict(self, as_string = False):
        return {
            "observation":      self.observation,
            "action":           self.action,
            "reward":           self.reward if as_string == False else str(self.reward),
            "next_observation": self.next_observation,
            "done":             self.done if as_string == False else str(self.done)
        }
    def __str__(self):
        printout = '\n'
        for key in self.as_dict():
            printout += "\n" + key + ':'
            printout += '\n' + str(self.as_dict()[key])
        return printout

class Trajectory:
    def __init__(self):
        self.transitions = []
        self.info        = {}
    def __len__(self):
        return len(self.transitions)
    def check_consistency(self):
        assert(any([transition.done for transition in self.transitions[:-1]]) == False) # should not be done until the end
        assert(self.transitions[ -1].done == True )
        for t in range(1,len(self.transitions)):
            prior_transition   = self.transitions[t-1]
            current_transition = self.transitions[t]
            assert(prior_transition.next_observation == current_transition.observation)
    def get_rewards(self):
        return [transition.reward for transition in self.transitions]
    def get_return(self):
        return sum([transition.reward for transition in self.transitions])
    def append(self, transition):
        assert(self.transitions == [] or self.transitions[-1].done == False)
        self.transitions.append(Transition(**transition))
    def __str__(self):
        printout = '\n*** Trajectory Begins *** \n'
        printout += "\nTrajectory Length: " + str(len(self))
        for idx, transition in enumerate(self.transitions):
            printout += "\nTransition: " + str(idx)
            printout += "\n" + transition.__str__()
        if self.info != None:
            printout += "\nFound Special Items"  
            printout += str(self.info)
        printout += '\n *** Trajectory Ends **** \n'
        return printout

class TrajectoryDataset(Dataset):
    def __init__(self):
        self.trajectories              = []
        self.samples                   = [] # pointer list for fast sampling
        self._last_oar                 = None # Last (observation, action, reward) for sequential addition
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx): 
        self.samples[idx].times_was_sampled += 1
        ### Must return a copy to avoid issues if further processing is done
        return copy.deepcopy(self.samples[idx].as_dict())
    def append_trajectory(self, trajectory: Trajectory):
        trajectory.check_consistency()
        assert(self.last_trajectory_reached_end())
        for transition in trajectory.transitions:
            self.append_sample_sequentially(copy.deepcopy(transition.as_dict()))
        self.trajectories[-1].info = copy.deepcopy(trajectory.info)
        self.trajectories[-1].check_consistency()
    def append_observation_action_reward(self, observation, action, reward):
        if self._last_oar != None:
            self.append_sample_sequentially({"observation": self._last_oar["observation"],
                                             "action": self._last_oar["action"],
                                             "reward": self._last_oar["reward"],
                                             "next_observation": observation,
                                             "done": False })
        self._last_oar = {"observation": observation,
                          "action": action,
                          "reward": reward}
    def append_terminal_observation(self, observation, trajectory_info = None):
        assert self._last_oar != None
        self.append_sample_sequentially({"observation": self._last_oar["observation"],
                                         "action": self._last_oar["action"],
                                         "reward": self._last_oar["reward"],
                                         "next_observation": observation,
                                         "done": True })
        self._last_oar = None
        if trajectory_info != None:
            self.trajectories[-1].info = trajectory_info
        self.trajectories[-1].check_consistency()

    def last_trajectory_reached_end(self):  
        return (self.trajectories == [] or self.trajectories[-1].transitions[-1].done)
         
    def append_sample_sequentially(self, transition):
        ### is the trajectory new?
        if self.last_trajectory_reached_end():
            self.trajectories.append(Trajectory())
        self.trajectories[-1].transitions.append(Transition(**transition))
        self.samples.append(self.trajectories[-1].transitions[-1])
    def nTrajectories(self):
        return len(self.trajectories)
    def get_all_trajectory_returns(self):
        return np.asarray([trajectory.get_return() for trajectory in self.trajectories])
    def check_consistency(self):
        assert (sum([len(trajectory) for trajectory in self.trajectories]) == len(self.samples))
        for trajectory in self.trajectories:
            trajectory.check_consistency()
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, len(self.samples), size=(batch_size,))
        # rand_indices = [np.random.randint(0, len(self.samples)) for _ in range(batch_size)]
        for idx in rand_indices:
            self.samples[idx].times_was_sampled += 1 
        return {
            "observation":      [self.samples[idx].observation            for idx in rand_indices],
            "action":           [self.samples[idx].action                 for idx in rand_indices],
            "reward":           [self.samples[idx].reward                 for idx in rand_indices],
            "next_observation": [self.samples[idx].next_observation       for idx in rand_indices],
            "done":             [self.samples[idx].done                   for idx in rand_indices],
        }
    def mean_trajectory_return(self):
        return np.mean(self.get_all_trajectory_returns())
    def std_trajectory_return(self):
        return np.std(self.get_all_trajectory_returns())
    def merge(self, dataset):
        self.check_consistency()
        dataset.check_consistency()
        for trajectory in dataset.trajectories:
            for transition in trajectory.transitions:
                self.append_sample_sequentially(transition.as_dict())
            self.trajectories[-1].info = copy.deepcopy(trajectory.info)
        self.check_consistency()
        # assert(self.batch_size == dataset.batch_size)
    def __str__(self):
        printout = '\n \n '
        printout += '\n ************************ '
        printout += '\n *** Printing Dataset *** '
        printout += '\n ************************ '
        printout += '\n \n '
        printout += '\n Number of Samples    : ' + str(len(self))
        printout += '\n Dataset Trajectories : ' + str(self.nTrajectories()) + '\n'
        for idx, trajectory in enumerate(self.trajectories):
            printout += "\n >>> Trajectory id: " + str(idx) + '\n'
            printout += trajectory.__str__()
        if self._last_oar != None:
            printout += "\n !!! Found incomplete transition !!! \n"
            for key in self._last_oar:
                printout += key + '\n'
                printout += str(self._last_oar[key]) + "\n"
        printout += '\n ************************ '
        printout += '\n *** Dataset  Printed *** '
        printout += '\n ************************ '
        return printout
    
    def keep_top_fraction_of_trajectories(self, fraction: float, from_high_to_low = True):
        self.sort(from_high_to_low=from_high_to_low)
        trajectories = self.trajectories
        import math
        nTraj_to_keep = int(fraction * self.nTrajectories())
        self.__init__()
        for i in range(nTraj_to_keep):
            self.append_trajectory(trajectories[i])
        print("*** Kept ", self.nTrajectories(), " trajectories")

    def keep_bottom_fraction_of_trajectories(self, fraction: float):
        self.keep_top_fraction_of_trajectories(fraction=fraction, from_high_to_low=False)


    def max_trajectory_return(self):
        return max(self.get_all_trajectory_returns())
    
    def argmax_trajectory_return(self):
        return np.argmax(self.get_all_trajectory_returns())
    
    def min_trajectory_return(self):
        return min(self.get_all_trajectory_returns())
    
    def argmin_trajectory_return(self):
        return np.argmin(self.get_all_trajectory_returns())
    
    def sort(self, from_high_to_low):
        print("Warning: new dataset created!")
        returns = [trajectory.get_return() for trajectory in self.trajectories]  
        sorted_trajectories = sort_list(self.trajectories, returns, from_high_to_low)
        self.__init__()
        for traj in sorted_trajectories:
            self.append_trajectory(traj)
        
    # useful to set all rewards to eg -1 and encourage reaching the goal faster
    def set_all_rewards_to_value(self, value):
        for sample in self.samples:
            sample.reward = np.single(value)
    
    def scale_all_rewards_by_value(self, value):
        for sample in self.samples:
            sample.reward *= np.single(value)

    def add_value_to_all_rewards(self, value):
        for sample in self.samples:
            sample.reward += np.single(value)

    def increase_final_reward_by_value(self, value):
        for trajectory in self.trajectories:
            trajectory.transitions[-1].reward  += np.single(value)
            
    def append_eos_token_to_all_actions(self, eos_token):
        for sample in self.samples:
            sample.action += eos_token
        
    def push_all_rewards_at_the_end_of_the_trajectory(self):
        for trajectory in self.trajectories:
            trajectory.transitions[-1].reward  = np.single(trajectory.get_return())
            for transition in trajectory.transitions[:-1]:
                transition.reward = np.single(0)
            assert(- len(trajectory) == trajectory.get_return() == trajectory.transitions[-1].reward)

    def save(self, filename):
        import json 
        self.check_consistency()
        with open(filename, "w") as final:
            json.dump([sample.as_dict(as_string = True) for sample in self.samples], final)

    def load(self, filename):
        import json 
        with open(filename, "r") as final:
            data = json.load(final)
            for sample in data:
                self.append_sample_sequentially(sample)
                
    def times_was_sampled(self):
        return [sample.times_was_sampled for sample in self.samples]
    
    def keep_only_trajectories_with_exact_key_and_value(self, key, value):
        trajectories = self.trajectories
        new_dataset = TrajectoryDataset()
        for trajectory in trajectories:
            if trajectory.info[key] == value:
                new_dataset.append_trajectory(trajectory)
        return new_dataset
    
    def construct_tabular_state_action_space(self):
        self.state_space        = Counter()
        self.action_space       = Counter()
        self.state_action_space = Counter()
        for sample in self.samples:
            self.state_space.add(sample.observation)
            self.action_space.add(sample.action)
            self.state_action_space.add((sample.observation, sample.action))

    def assert_deterministic(self):
        successor_states = {}
        rewards          = {}
        for sample in self.samples:
            sa = (sample.observation, sample.action)
          
            if sa not in rewards:
                rewards[sa] = sample.reward

            else:
                assert(rewards[sa] == sample.reward)
            
            if sample.done: # end transition may be ill-defined
                continue

            if sa not in successor_states:
                successor_states[sa] = sample.next_observation
            else:
                assert(successor_states[sa] == sample.next_observation)
        
class Counter():
    def __init__(self):
        self.register = {}
    def add(self, item):
        if item not in self.register:
            self.register[item] = 1
        else:
            self.register[item] += 1
    def contains(self, item):
        return item in self.register

    def n_samples(self, item):
        return self.register[item]
    
class EmptyDataset():
    def __init__(self, length):
        self.length     = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [0]
    
def sort_list(list1, list2, from_high_to_low):
    # Sorting the List1 based on List2
    return [val for (_, val) in sorted(zip(list2, list1), key=lambda x: x[0], reverse=from_high_to_low)]
