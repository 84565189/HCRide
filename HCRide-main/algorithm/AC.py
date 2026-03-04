import collections
import random
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda")
print(f"Using device: {device}")  # 验证是否识别到 GPU

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim,actionDim):  # 58 # 8
        super(PolicyNet, self).__init__()
        self.stateDim = stateDim # 58
        self.actionDim = actionDim # 8
        self.S = torch.nn.Linear(self.stateDim, 32)
        self.A = torch.nn.Linear(self.actionDim,4)
        self.L1 = torch.nn.Linear(32+4,16)
        self.L2 = torch.nn.Linear(16,8)
        self.f = torch.nn.Linear(8, 1)

    def forward(self, X):
        s = X[:, :self.stateDim]
        a = X[:, -self.actionDim:]
        s1 = F.relu(self.S(s))
        a1 = F.relu(self.A(a))
        y1 = torch.cat((s1, a1), dim=1)
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return self.f(l2)


# critic
class RewardValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(RewardValueNet, self).__init__()
        self.stateDim = stateDim  # 58
        self.S = torch.nn.Linear(stateDim, 48)
        self.L1 = torch.nn.Linear(48, 32)
        self.L2 = torch.nn.Linear(32, 8)
        self.f = torch.nn.Linear(8, 1)

    def forward(self, X):
        s = X[:, :self.stateDim]
        y1 = F.relu(self.S(s))
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return F.relu(self.f(l2))

class ReplayBuffer:
    def __init__(self, capacity,batchSize):
        self.buffer = collections.deque(maxlen=capacity)
        self.batchSize = batchSize

    def add(self,matchingState,state,action,reward,cost,nextState):
        self.buffer.append((matchingState,state,action,reward,cost,nextState))

    def sample(self):
        transitions = random.sample(self.buffer, self.batchSize)
        matchingState,state,action,reward,cost,nextState = zip(*transitions)
        state = list(state)
        state = [x.tolist() for x in state]
        return matchingState,state, action, reward,cost,nextState

    def size(self):
        return len(self.buffer)

class AC:
    def __init__(self, stateDim, actionDim, actorLr, criticLr,lagLr,limit,lagrange,epochs,eps, gamma, batchSize):
        self.actor = PolicyNet(stateDim,actionDim)
        self.rewardCritic = RewardValueNet(stateDim)
        self.actorLr = actorLr
        self.rewardCriticLr = criticLr # update step for reward
        self.costCriticLr = criticLr # update step for cost
        self.lagLr = lagLr
        self.limit = limit
        self.gamma = gamma  # discount factor
        self.lagrange = torch.tensor(lagrange,dtype=torch.float,requires_grad=True)
        self.epochs = epochs  # round
        self.eps = eps   # clip range
        self.batchSize = batchSize
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actorLr)
        self.rewardCriticOptimizer = torch.optim.Adam(self.rewardCritic.parameters(), lr=self.rewardCriticLr)

    def take_action(self, matchingState,dayIndex):
        matchingState = torch.tensor(matchingState,dtype=torch.float)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)
        actionProb = torch.softmax(vOutput, dim=0)
        a = random.random()
        if a < 1:
            actionDist = torch.distributions.Categorical(actionProb)
            action = actionDist.sample().cpu()
        else:
            action = torch.max(actionProb,0)[1]
        return action.item()


    def take_action_execute(self, matchingState,dayIndex):
        matchingState = torch.tensor(matchingState,dtype=torch.float)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)
        actionProb = torch.softmax(vOutput, dim=0)
        a = random.random()
        if a < 0:
            actionDist = torch.distributions.Categorical(actionProb)
            action = actionDist.sample().cpu()
        else:
            action = torch.max(actionProb,0)[1]
        return action.item()
    def update_theta(self, matchingState,state,action,reward,cost,nextState,round):
        state = torch.tensor(state, dtype=torch.float) # n * 244
        action = torch.tensor(action).view(-1,1)# n * 1
        reward = torch.tensor(reward, dtype=torch.float).view(-1,1) # n * 1
        cost = torch.tensor(cost, dtype=torch.float).view(-1,1) # n * 1
        nextState = torch.tensor(nextState, dtype=torch.float) # n * 244

        rewardTarget = reward + self.gamma * self.rewardCritic(nextState)
        rewardAdvantage = rewardTarget - self.rewardCritic(state)

        oldLogProb = torch.tensor([])
        for i in range(self.batchSize):
            matchingStateOne = torch.tensor(matchingState[i], dtype=torch.float)
            lP = torch.log(torch.softmax(self.actor(matchingStateOne), dim=0)[action[i].item()])
            oldLogProb = torch.cat((oldLogProb,lP),0)

        # if round % 5 == 0:
        #     self.reset_critic_learning_rate()
        # if round % 20 == 0:
        #     self.reset_actor_learning_rate()


        actorLoss = torch.mean(-oldLogProb * rewardAdvantage.detach())
        rewardCriticLoss = torch.mean(F.mse_loss(self.rewardCritic(state), rewardTarget.detach()))

        self.actorOptimizer.zero_grad()
        self.rewardCriticOptimizer.zero_grad()
        actorLoss.backward()
        rewardCriticLoss.backward()
        # print(f'actorLoss:{actorLoss}')
        # print(f'rewardLoss:{rewardCriticLoss}')
        self.actorOptimizer.step()
        self.rewardCriticOptimizer.step()


    def reset_critic_learning_rate(self):
        self.rewardCriticLr = self.rewardCriticLr / 3
        #self.actorLr = self.actorLr
        self.rewardCriticOptimizer.param_groups[0]['lr'] = self.rewardCriticLr
        #self.actorOptimizer.param_groups[0]['lr'] = self.actorLr

    def reset_actor_learning_rate(self):
        self.actorLr = self.actorLr / 2
        self.actorOptimizer.param_groups[0]['lr'] = self.actorLr