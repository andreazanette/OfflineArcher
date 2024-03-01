import lightning as L
import torch
torch.set_float32_matmul_precision('high')

from SimulateOnEnv import batch_simulate_on_environment
from lightning.pytorch.callbacks import Callback

class GPT2(torch.nn.Module):
    def __init__(self, get_device, from_checkpoint = None):
        super().__init__()

        self.get_device = get_device

        ### Initialize Model
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        if from_checkpoint is not None:
            checkpoint = torch.load(from_checkpoint, map_location = torch.device('cpu'))
            weights = {k.removeprefix("agent."): v for k, v in checkpoint["state_dict"].items() if k.startswith("agent.")}
            self.load_state_dict(weights)
            print("I have initialized the actor from the checkpoint: ", from_checkpoint)

        ### Initialize Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        self.tokenizer.truncation_side = 'left' # Truncation happens during generation and computation of log probabilities
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def forward(self, observation, do_sample = True):
        obs_ids    = self.tokenizer(observation, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.model.device)
        obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"])
        outputs = self.model.generate(inputs_embeds=obs_embeds, attention_mask=obs_ids['attention_mask'],\
                                       max_new_tokens=32, do_sample=do_sample, \
                                       pad_token_id = self.tokenizer.eos_token_id)
        action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        return action

    def behavioral_cloning_loss(self, observation, action, **kwargs):
        logsum_probs = self.get_logsum_prob(observation, action) # this line has been refactored and not tested
        loss = - logsum_probs.mean()
        return loss, {"behavioral_cloning/loss": loss.detach()}

    def get_logsum_prob(self, observation, action_from_dataloader, **kwargs):
        action = [a + self.tokenizer.eos_token for a in action_from_dataloader]
        alltext  =  [obs + a for obs, a in zip(observation, action)]
        generated_probabilities = self.to_tokens_and_logprobs(alltext)
        assert(len(generated_probabilities) == len(alltext)  == len(observation) == len(action))
        mask = torch.zeros_like(generated_probabilities.detach(),dtype=torch.bool)

        for i, (obs, act, text) in enumerate(zip(observation, action, alltext)):
            assert(text == obs+act)
            act_ids = self.tokenizer(act,  return_tensors='pt', padding=True)
            txt_ids = self.tokenizer(text, return_tensors='pt', padding=True)
            n_token_act   = len(act_ids["input_ids"][0]) # [0] because the batch is one inside the foor loop
            n_token_txt   = len(txt_ids["input_ids"][0])
            mask[i, n_token_txt - n_token_act -1 : n_token_txt -1] = True # the -1 shift is due to the the generated probabilities being shifted

        generated_probabilities = torch.where(mask, generated_probabilities, 1.0)
        log_probs               = torch.where(mask, torch.log(generated_probabilities), 0.0) # must be separate from the line above for numerical stability (cannot take log(0.0))
        logsum_probs            = torch.sum(log_probs, dim = 1)
        return logsum_probs

    def to_tokens_and_logprobs(self, input_texts):
        input_ids = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.get_device())
        outputs = self.model(input_ids)
        probs = torch.softmax(outputs.logits, dim=-1)

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        return gen_probs
    
class RobertaCritic(torch.nn.Module):
    def __init__(self, get_device, discount_factor: float, tau: float, expectile: float, from_checkpoint=None):
        super().__init__()

        self.get_device      = get_device
        self.discount_factor = discount_factor
        self.tau             = tau
        self.expectile       = expectile

        ### Define the Critic
        from ArcherCritic import ArcherDoubleCritic
        self.critic          = ArcherDoubleCritic(in_dim = 768, out_dim = 1)  
        self.target_critic   = ArcherDoubleCritic(in_dim = 768, out_dim = 1) 
        self.soft_update_target_critic(1)

        if from_checkpoint is not None:
            checkpoint = torch.load(from_checkpoint, map_location = torch.device('cpu'))
            weights = {k.removeprefix("critic."): v for k, v in checkpoint["state_dict"].items() if k.startswith("critic.")}
            self.load_state_dict(weights)
            print("I have initialized the critic from the checkpoint: ", from_checkpoint)

        ### Miscellaneus Shortcuts
        self.softmax             = torch.nn.Softmax(dim= -1)
        self.td_criterion        = torch.nn.MSELoss()
        self.expectile_criterion = lambda diff: self.loss_value_diff(diff = diff, expectile = self.expectile)

    def get_q(self, observation, action, detach_model=False):
        return self.critic.get_q(observation, action, detach_model = detach_model)

    def get_v(self, inputs, detach_model=False):
        return self.critic.get_v(inputs, detach_model = detach_model)

    def get_target_v(self, inputs, detach_model=False):
        return self.target_critic.get_v(inputs, detach_model = detach_model)
    
    def get_target_q(self, observation, action, detach_model=False):
        return self.target_critic.get_q(observation, action, detach_model = detach_model)

    def get_advantages(self, observation, action):
            q1, q2 = self.get_q(observation, action)
            v1, v2 = self.get_v(observation) 
            q = torch.minimum(q1, q2)
            v = torch.minimum(v1, v2)
            advantages = q - v
            return advantages
    
    def argmax_advantage(self, observation, get_available_actions):
        argmax_actions = []
        for obs in observation:
            available_actions = get_available_actions(obs)
            advantages = torch.as_tensor([self.get_advantages([obs], [action]) for action in available_actions])
            action = available_actions[torch.argmax(advantages)]
            argmax_actions.append(action)
        return argmax_actions
    
    def soft_update_target_critic(self, tau = None):
        if tau == None:
            tau = self.tau
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
    
    def iql_loss(self, observation, action, reward, next_observation, done, **kwargs):
        ### Fitting the Q function
        q1, q2 = self.get_q(observation, action, detach_model=False)
        q1 = q1.flatten()
        q2 = q2.flatten()

        reward = torch.Tensor(reward)#.to(self.agent.device)
        done = torch.Tensor(done)#.to(self.agent.device)
        
        with torch.no_grad():
            target_v1, target_v2 = self.get_target_v(next_observation)
           
            target_v1 = reward + torch.logical_not(done)*target_v1.flatten()*self.discount_factor
            target_v2 = reward + torch.logical_not(done)*target_v2.flatten()*self.discount_factor
            

        q1_loss = self.td_criterion(q1, target_v1)
        q2_loss = self.td_criterion(q2, target_v2)

        ### Fitting the value function
        with torch.no_grad():
            target_q1, target_q2 = self.get_target_q(observation, action, detach_model=False)
        target_q1 = target_q1.flatten()
        target_q2 = target_q2.flatten()

        v1, v2 = self.get_v(observation, detach_model=False)
        v1 = v1.flatten()
        v2 = v2.flatten()

        v1_loss = self.expectile_criterion(diff = target_q1.detach() - v1)
        v2_loss = self.expectile_criterion(diff = target_q2.detach() - v2)

        loss = q1_loss + q2_loss + v1_loss + v2_loss

        ### Log and print what's happening
        log = self.get_log(q1=q1, q2=q2, v1=v1, v2=v2, q1_loss=q1_loss, q2_loss=q2_loss, v1_loss=v1_loss, v2_loss=v2_loss, target_q1=target_q1, target_q2=target_q2)
        return loss, log
    
    def loss_value_diff(self, diff, expectile):
        """Loss function for iql expectile value difference."""
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return (weight * (diff**2)).mean()  
    
    def get_log(self, q1, q2, v1, v2, q1_loss, q2_loss, v1_loss, v2_loss, target_q1, target_q2):
        return {"critic/q1.loss": q1_loss.detach(),\
                    "critic/q2.loss": q2_loss.detach(),\
                    "critic/v1.loss": v1_loss.detach(),\
                    "critic/v2.loss": v2_loss.detach(),\
                    "critic/q1.mean": torch.mean(q1).detach(),\
                    "critic/q1.min": torch.min(q1).detach(),\
                    "critic/q1.max": torch.max(q1).detach(),\
                    "critic/q2.mean": torch.mean(q2).detach(),
                    "critic/q2.max": torch.max(q2).detach(),
                    "critic/q2.min": torch.min(q2).detach(),
                    "critic/v1.mean": torch.mean(v1).detach(),\
                    "critic/v1.min": torch.min(v1).detach(),\
                    "critic/v1.max": torch.max(v1).detach(),\
                    "critic/v2.mean": torch.mean(v2).detach(),
                    "critic/v2.max": torch.max(v2).detach(),
                    "critic/v2.min": torch.min(v2).detach(),
                    "critic/target_q1.mean": torch.mean(target_q1).detach(),\
                    "critic/target_q1.min": torch.min(target_q1).detach(),\
                    "critic/target_q1.max": torch.max(target_q1).detach(),\
                    "critic/target_q2.mean": torch.mean(target_q2).detach(),
                    "critic/target_q2.max": torch.max(target_q2).detach(),
                    "critic/target_q2.min": torch.min(target_q2).detach()}

class Agent(L.LightningModule):
    def validation_step(self, batch, batch_idx):

        # Perform evaluation on environment with stochastic policy
        eval_dataset               = batch_simulate_on_environment(policy = lambda obs: self.forward(obs),                  env = self.trainer.datamodule.env) 
        self.log("eval/avg_return", eval_dataset.mean_trajectory_return(), sync_dist = True)
        self.log("eval/std_return", eval_dataset.std_trajectory_return(),  sync_dist = True)
        
        # Perform evaluation on environment with deterministic policy
        deterministic_eval_dataset = batch_simulate_on_environment(policy = lambda obs: self.forward(obs, do_sample=False), env = self.trainer.datamodule.env) 
        self.log("eval/avg_return_deterministic", deterministic_eval_dataset.mean_trajectory_return(), sync_dist = True)
        self.log("eval/std_return_deterministic", deterministic_eval_dataset.std_trajectory_return(),  sync_dist = True)
        
        return eval_dataset.mean_trajectory_return()
    
class BehaviouralCloning(Agent):
    def __init__(self, lr: float):
        super().__init__() # Initialize LLM base class
        self.save_hyperparameters()

        ### Config
        self.lr    = lr

        ### Initialization
        self.agent = GPT2(get_device = lambda: self.device)

    def forward(self, observation, **kwargs):
        return self.agent.forward(observation, **kwargs)

    def training_step(self, batch, batch_idx):
        loss, log = self.agent.behavioral_cloning_loss(**batch)
        self.log_dict(log, sync_dist=True)  
        return loss
    
    def configure_optimizers(self):
        from torch.optim import Adam
        optimizer = Adam(self.agent.model.parameters(), lr = self.lr)
        return optimizer

class FilteredBehaviouralCloning(BehaviouralCloning):
    def __init__(self, lr: float, filter: float):
        super().__init__(lr) 

        self.filter = filter

    def configure_callbacks(self):
        return FilterDataset(filter = self.filter)

class FilterDataset(Callback):
    def __init__(self, filter: float):
        self.filter = filter

    def on_fit_start(self, trainer, algorithm):
        print("*** Filtering Dataset ***")
        dataset = trainer.datamodule.dataset
        print("Statistics of Input Dataset")
        print("Number of Trajectories:", dataset.nTrajectories())
        print("Number of Trajectories:", len(dataset))
        dataset.keep_top_fraction_of_trajectories(fraction=self.filter)
        trainer.datamodule.dataset = dataset
        print("Statistics of Filtered Dataset")
        print("Number of Trajectories:", dataset.nTrajectories())
        print("Number of Trajectories:", len(dataset))

class ActorCritic(Agent):
    def __init__(self, actor_lr: float, critic_lr: float, tau: float, accumulate_grad_batches: int, discount_factor: float, critic_expectile: float, optimize_critic: bool, actor_checkpoint=None, critic_checkpoint=None, **kwargs):
        super().__init__() # Initialize LLM base class
        self.save_hyperparameters()

        ### Config
        self.actor_lr                = actor_lr
        self.critic_lr               = critic_lr
        self.discount_factor         = discount_factor
        self.tau                     = tau

        ### Manual Gradient Accumulation
        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization  = False

        ### Initialization
        self.actor  = GPT2         (get_device = lambda: self.device, from_checkpoint=actor_checkpoint)
        self.critic = RobertaCritic(get_device = lambda: self.device, discount_factor = discount_factor, tau = tau, expectile = critic_expectile, from_checkpoint=critic_checkpoint)

        self.actor_current_backward_step  = 0
        self.critic_current_backward_step = 0
        self.critic_warmup_gradient_steps = 0

        self.optimize_actor          = lambda: True if self.critic_current_backward_step // self.accumulate_grad_batches >= self.critic_warmup_gradient_steps else False
        self.optimize_critic         = lambda: optimize_critic 
        
    def forward(self, observation, **kwargs):
        action = self.actor.forward(observation, **kwargs)
        return action

    def training_step(self, batch, batch_idx):
        actor_optimizer, critic_optimizer = self.optimizers()

        if self.optimize_critic():
            # scale losses by 1/N (for N batches of gradient accumulation)
            critic_loss, critic_log = self.critic_loss(batch)            
            critic_loss /= self.accumulate_grad_batches
            self.manual_backward(critic_loss)
            self.critic_current_backward_step += 1
            self.log_dict(critic_log, sync_dist=True)

            # accumulate gradients of N batches
            if self.critic_current_backward_step % self.accumulate_grad_batches == 0:
                critic_optimizer.step()
                critic_optimizer.zero_grad()
                self.critic.soft_update_target_critic(self.tau)

        if self.optimize_actor():
            # scale losses by 1/N (for N batches of gradient accumulation)
            actor_loss, actor_log  = self.actor_loss(batch) 
            actor_loss /= self.accumulate_grad_batches
            self.manual_backward(actor_loss)
            self.actor_current_backward_step += 1
            self.log_dict(actor_log, sync_dist=True)

            # accumulate gradients of N batches
            if self.actor_current_backward_step % self.accumulate_grad_batches == 0:
                actor_optimizer.step()
                actor_optimizer.zero_grad()
    
    def get_actor_log(self, loss, advantages, log_prob):
        return  {"actor/loss": loss.detach(),\
            "actor/advantages.mean": advantages.detach().mean(),\
            "actor/advantages.max": torch.max(advantages.detach()),\
            "actor/advantages.min": torch.min(advantages.detach()),
            "actor/log_prob.mean": torch.mean(log_prob.detach()),
            "actor/log_prob.max": torch.max(log_prob.detach()),
            "actor/log_prob.min": torch.min(log_prob.detach()),
            } 
    
    def configure_optimizers(self):
        from torch.optim import Adam
        actor_optimizer  = Adam(self.actor.model.parameters(),  lr = self.actor_lr)
        critic_optimizer = Adam(self.critic.critic.parameters(), lr = self.critic_lr) # access the critic, not the target critic
        return actor_optimizer, critic_optimizer


class OfflineArcher(ActorCritic):
    def __init__(self, inv_temp: float, **kwargs):
        super().__init__(**kwargs)

        self.inv_temp = inv_temp
        
        self.actor_loss  = lambda batch: self.awr_loss(**batch)
        self.critic_loss = lambda batch: self.critic.iql_loss(**batch)

    def awr_loss(self, observation, action, **kwargs):
        log_prob    = self.actor.get_logsum_prob(observation, action)
        with torch.no_grad():
            advantages = self.critic.get_advantages(observation, action)
        
        advantages = advantages.flatten()
        log_prob   = log_prob.flatten()
        factor     = torch.exp(self.inv_temp * advantages)
        loss_batch = -factor*log_prob
        loss       = loss_batch.mean()

        # ### Log and print what's happening
        log = self.get_actor_log(loss = loss, advantages = advantages, log_prob = log_prob)
        log = {**log, **{ "actor/factor.mean": factor.detach().mean(),\
            "actor/factor.max": torch.max(factor.detach()),\
            "actor/factor.min": torch.min(factor.detach())}
            }

        return loss, log     
    
class IQLKL(ActorCritic):
    def __init__(self, kl_coeff: float, reference_actor_path, **kwargs):
        super().__init__(**kwargs)

        self.kl_coeff = kl_coeff
        self.reference_actor = GPT2(get_device = lambda: self.device, from_checkpoint = reference_actor_path)
        
        self.actor_loss  = lambda batch: self.advantage_kl_loss(**batch)
        self.critic_loss = lambda batch: self.critic.iql_loss(**batch)

    def advantage_kl_loss(self, observation, **kwargs):
        reinforce_loss, generated_output = self.reinforce_loss(observation=observation)
        with torch.no_grad():
            reference_log_prob = self.reference_actor.get_logsum_prob(observation, generated_output["action"])

        ratio   = generated_output["log_prob"] - reference_log_prob
        kl_loss = (ratio.detach() + 1.0)*generated_output["log_prob"]
        loss = (1-self.kl_coeff) * reinforce_loss + self.kl_coeff * kl_loss
        log = generated_output["log"]
        log = {**log, "reference_log_prob.mean": reference_log_prob.mean(), "reference_log_prob.max": reference_log_prob.max(), "reference_log_prob.min": reference_log_prob.min()}
        log = {**log, "kl_loss.mean": kl_loss.mean(), "kl_loss.max": kl_loss.max(), "kl_loss.min": kl_loss.min()}
        log = {**log, "actor_loss.mean": loss.mean(), "ratio": ratio.mean()}
        
        return loss.mean(), log

    def reinforce_loss(self, observation, **kwargs):
        ### Reinforce Loss
        action      = self.actor.forward(observation)
        log_prob    = self.actor.get_logsum_prob(observation, action)

        with torch.no_grad():
            advantages = self.critic.get_advantages(observation, action)

        loss = -advantages.flatten()*log_prob

        ### Logging
        log = self.get_actor_log(loss = torch.mean(loss.detach()), advantages = advantages, log_prob = log_prob)
        # self.log_dict(log)
        return loss, {"log_prob": log_prob, "advantages": advantages, "action": action, "log": log}
