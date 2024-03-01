from lightning import LightningDataModule
import torch.utils.data as data
from Dataset import TrajectoryDataset, EmptyDataset
from SimulateOnEnv import batch_simulate_on_environment
import numpy as np
from copy import deepcopy
import sys

class Task(LightningDataModule):
    def __init__(self, batch_size: int, n_traj_eval: int, **kwargs):
        super().__init__(**kwargs)
        self.batch_size                 = batch_size
        self.eval_batch_size            = self.batch_size
        self.n_traj_eval                = n_traj_eval

        # Set Defaults
        self.shuffle                    = True
        self.drop_last                  = True # skips last batch to make sure gradient accumulation works as intended

    def setup(self, stage: str):
        raise NotImplementedError
    
    def train_dataloader(self):
        return data.DataLoader(dataset = self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last = self.drop_last)

    def val_dataloader(self):
        return data.DataLoader(dataset = EmptyDataset(length = self.n_traj_eval), batch_size=self.eval_batch_size)

    def get_eval_log(self, **kwargs):
        pass

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
    
class TwentyQuestions(Task):
    def __init__(self, word_list = None, **kwargs):
        super().__init__(**kwargs)

        self.word_list                  = word_list
        self.max_horizon                = 20
        
        from twenty_questions import BatchedTwentyQuestionsEnv
        self.env    = BatchedTwentyQuestionsEnv(max_conversation_length = self.max_horizon, bsize = self.eval_batch_size, word_list=self.word_list)

    def setup(self, stage: str):
        self.env.model.to(self.trainer.model.device) # Ensure environment is on GPU
        self.dataset = self.read_data()
        self.dataset.check_consistency()
        print("\n *** Dataset Trimming Now Disabled. Please Called the Subroutine for triming")

    def read_data(self):
        import json
        from Dataset import TrajectoryDataset

        f = open('datasets/20q_train.json')
        data    = json.load(f)
        dataset              = TrajectoryDataset()

        for game in data:
            assert(len(game['lines']) <= 20)
            history = "Questions:\n" # assertion is checked with history = ''
            for interaction in game['lines']:
                yesAnswer = interaction[-5:] == ' Yes.'
                noAnswer  = interaction[-4:] == ' No.' 
                assert(yesAnswer or noAnswer)
                observation  = history
                
                done = True if interaction == game['lines'][-1] else False # if the interaction is the last interaction we are done
                reward = 0 if done and game['correct'] else -1
                
                if yesAnswer:
                    action = interaction[:-5]
                if noAnswer:
                    action = interaction[:-4]

                history += interaction + '\n'
                dataset.append_observation_action_reward(observation, action, reward)
            dataset.append_terminal_observation(history, 
                                                trajectory_info = {"correct": game["correct"], 
                                                                "word": game["word"]})

        dataset.check_consistency()
        return dataset 