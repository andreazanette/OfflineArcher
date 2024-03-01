import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import logging
# A logger for this file
log = logging.getLogger(__name__)

class ArcherDoubleCritic(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ArcherDoubleCritic, self).__init__()
        self.base_lm = RobertaModel.from_pretrained('roberta-base')

        ################
        print("*** Master Warning - Are these used? *** ")
        # self.base_lm.pooler.dense.weight = None
        # self.base_lm.pooler.dense.bias   = None
        ###############
        self.base_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.base_tokenizer.truncation_side = 'left'
        self.critic1 = nn.Sequential(nn.Linear(in_dim*2, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
        self.critic2 = nn.Sequential(nn.Linear(in_dim*2, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
        self.v_critic1 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
        self.v_critic2 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
    
    def get_q(self, observation, action, detach_model=False):
        state_actions = [o + a for o,a in zip(observation, action)]
        obs_ids = self.base_tokenizer(observation, padding = True, return_tensors='pt', truncation=True, max_length=512).to(self.base_lm.device)
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).last_hidden_state[:,0]
        else:
            lm_states = self.base_lm(**obs_ids).last_hidden_state[:,0]
        action_ids = self.base_tokenizer(action, padding = True, return_tensors='pt', truncation=True, max_length=512).to(self.base_lm.device)
        if detach_model:
            with torch.no_grad():
                action_states = self.base_lm(**action_ids).last_hidden_state[:,0]
        else:
            action_states = self.base_lm(**action_ids).last_hidden_state[:,0]
        lm_states = torch.cat([lm_states, action_states], dim = 1)
        return self.critic1(lm_states), self.critic2(lm_states)

    def get_v(self, observation,detach_model=False):
        obs_ids = self.base_tokenizer(observation, padding = True, return_tensors='pt', truncation=True, max_length=512).to(self.base_lm.device)
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).last_hidden_state[:,0]
        else:
            lm_states = self.base_lm(**obs_ids).last_hidden_state[:,0]
        # print(action.size())
        return self.v_critic1(lm_states), self.v_critic2(lm_states)


class ArcherCritic(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=4096, dropout = 0.5):
        super(ArcherCritic, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained('gpt2')
        self.critic = ArcherDoubleCritic(in_dim = 768, out_dim = 1)  
        self.target_critic = ArcherDoubleCritic(in_dim = 768, out_dim = 1) 
        self.soft_update_target_critic(1)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)

    def get_action(self, observation):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512).to(self.model.device)
        obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"])
        outputs = self.model.generate(inputs_embeds=obs_embeds, attention_mask=obs_ids['attention_mask'],\
                                       max_new_tokens=32, do_sample=True, \
                                       pad_token_id = self.tokenizer.eos_token_id)#.cpu()
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        return raw_action

    def get_q(self, observation, action, detach_model=False):
        return self.critic.get_q(observation, action, detach_model = detach_model)

    def get_v(self, inputs, detach_model=False):
        return self.critic.get_v(inputs, detach_model = detach_model)
    
    def get_target_q(self, observation, action, detach_model=False):
        return self.target_critic.get_q(observation, action, detach_model = detach_model)

    def get_log_prob(self, observation, action):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512).to(self.model.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512).to(self.model.device)
        action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
                                dim = 1)
        outputs = self.model(inputs_embeds = input_embeds, attention_mask = attention_mask)
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        logsum_probs = torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)
        return logsum_probs
    
    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

class DoubleCritic(torch.nn.Module):
    """
    a double critic without base lm
    """
    def __init__(self, in_dim, out_dim):
        super(DoubleCritic, self).__init__()
        # self.device = device
        self.critic1 = nn.Sequential(nn.Linear(in_dim*2, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
        self.critic2 = nn.Sequential(nn.Linear(in_dim*2, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
        self.v_critic1 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
        self.v_critic2 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))#.to(device)
    
    def get_q(self, observation, action, detach_model=False):
        lm_states = torch.cat([observation, action], dim = 1)
        return self.critic1(lm_states), self.critic2(lm_states)

    def get_v(self, observation,detach_model=False):
        return self.v_critic1(observation), self.v_critic2(observation)
