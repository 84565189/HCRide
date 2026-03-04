import collections
import random
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda")
print(f"Using device: {device}")  # 验证是否识别到 GPU

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim,actionDim):
        super(PolicyNet, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.S = torch.nn.Linear(self.stateDim,4)
        self.A = torch.nn.Linear(self.actionDim,4)
        self.L1 = torch.nn.Linear(4+4,6)
        self.L2 = torch.nn.Linear(6,4)
        self.f = torch.nn.Linear(4,1)

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

# torch.tanh
class RewardValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(RewardValueNet, self).__init__()
        self.stateDim = stateDim
        self.S = torch.nn.Linear(stateDim, 16)
        self.L1 = torch.nn.Linear(16, 8)
        self.L2 = torch.nn.Linear(8, 4)
        self.f = torch.nn.Linear(4, 1)

    def forward(self, X):
        s = X[:, :self.stateDim]
        y1 = F.relu(self.S(s))
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return self.f(l2)


class CostValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(CostValueNet, self).__init__()
        self.stateDim = stateDim  # 14
        self.S = torch.nn.Linear(stateDim, 16)
        self.L1 = torch.nn.Linear(16,8)
        self.L2 = torch.nn.Linear(8,4)
        self.f = torch.nn.Linear(4,1)

    def forward(self, X):
        s = X[:, :self.stateDim]
        y1 = F.relu(self.S(s))
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return self.f(l2)


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


class Habic:
    def __init__(self, stateDim, actionDim, actorLr, criticLr,lagLr,limit,lagrange,epochs,eps, gamma, batchSize):
        self.actor = PolicyNet(stateDim,actionDim)
        self.rewardCritic = RewardValueNet(stateDim)
        self.costCritic = CostValueNet(stateDim)
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
        self.costCriticOptimizer = torch.optim.Adam(self.costCritic.parameters(), lr=self.costCriticLr)
        self.lagOptimizer = torch.optim.SGD([self.lagrange], lr=self.lagLr)

    def take_action(self, matchingState,dayIndex):  # 训练
        matchingState = torch.tensor(matchingState,dtype=torch.float)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)  # 将二维张量变为一维张量
        actionProb = torch.softmax(vOutput, dim=0)
        actionDist = torch.distributions.Categorical(actionProb)
        action = actionDist.sample().cpu()
        return action.item()  # 对softmax函数求导时的用法

    def take_action_test(self, matchingState,dayIndex):  # 训练
        matchingState = torch.tensor(matchingState,dtype=torch.float)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)  # 将二维张量变为一维张量
        actionProb = torch.softmax(vOutput, dim=0)
        action = torch.max(actionProb,0)[1]
        return action.item()  # 对softmax函数求导时的用法

    def update_theta(self, matchingState,state,action,reward,cost,nextState,round,update_k):
        state = torch.tensor(state, dtype=torch.float) # n * 244
        action = torch.tensor(action).view(-1,1)# n * 1
        reward = torch.tensor(reward, dtype=torch.float).view(-1,1) # n * 1
        cost = torch.tensor(cost, dtype=torch.float).view(-1,1) # n * 1
        nextState = torch.tensor(np.array(nextState), dtype=torch.float) # n * 244


        nextRewardCritic = self.rewardCritic(nextState)
        rewardCritic = self.rewardCritic(state)
        nextCostCritic = self.costCritic(nextState)
        costCritic = self.costCritic(state)

        rewardTarget = reward + self.gamma * nextRewardCritic
        rewardAdvantage = rewardTarget - rewardCritic
        costTarget = cost + self.gamma * nextCostCritic
        costAdvantage = costTarget - costCritic


        oldLogProb = []
        for i in range(self.batchSize):
            matchingStateOne = torch.tensor(matchingState[i], dtype=torch.float)
            lPOne = torch.log(torch.softmax(self.actor(matchingStateOne), dim=0)[action[i].item()]).detach()
            oldLogProb.append(lPOne)


        if (round % 50 == 0):
            self.reset_reward_critic_learning_rate()
            self.reset_cost_critic_learning_rate()
        if (round % 50 == 0):
            self.reset_actor_learning_rate()


        for k in range(self.epochs):
            newLogProb = []
            for i in range(self.batchSize):
                matchingStatetwo = torch.tensor(matchingState[i], dtype=torch.float)
                lP = torch.log(torch.softmax(self.actor(matchingStatetwo), dim=0)[action[i].item()])
                newLogProb.append(lP)
            minRewardList = []
            minCostList = []
            for i in range(self.batchSize):
                ratio = torch.exp(newLogProb[i] - oldLogProb[i])
                rewardSurr1 = ratio * rewardAdvantage[i].detach()
                rewardSurr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * rewardAdvantage[i].detach()
                costSurr1 = ratio * costAdvantage[i].detach()
                costSurr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * costAdvantage[i].detach()
                minRewardSurr = torch.min(rewardSurr1,rewardSurr2)
                minCostSurr = torch.min(costSurr1, costSurr2)
                minRewardList.append(minRewardSurr)
                minCostList.append(minCostSurr)
            JR = torch.mean(torch.stack(minRewardList, dim=0), dim=0)
            JC = torch.mean(torch.stack(minCostList, dim=0), dim=0)
            L = JR


            actorLoss = -L
            rewardCriticLoss = torch.mean(F.mse_loss(self.rewardCritic(state), rewardTarget.detach()))
            costCriticLoss = torch.mean(F.mse_loss(self.costCritic(state), costTarget.detach()))

            if (update_k == 0) & (k == self.epochs - 1):
                # print(f'ActorLoss:{actorLoss.item()}')
                # print(f'CriticLoss:{rewardCriticLoss}')
                a,b,c = actorLoss.detach(), rewardCriticLoss.detach(),costCriticLoss.detach()

            self.actorOptimizer.zero_grad()
            self.rewardCriticOptimizer.zero_grad()
            self.costCriticOptimizer.zero_grad()
            actorLoss.backward()
            rewardCriticLoss.backward()
            costCriticLoss.backward()
            self.actorOptimizer.step()
            self.rewardCriticOptimizer.step()
            self.costCriticOptimizer.step()

        return a,b,c


    def update_lagrange(self, matchingState,state,action,reward,cost,nextState,round):
        state = torch.tensor(state, dtype=torch.float)  # n * 244
        action = torch.tensor(action).view(-1, 1)  # n * 1
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1)  # n * 1
        cost = torch.tensor(cost, dtype=torch.float).view(-1, 1)  # n * 1
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)  # n * 244

        rewardTarget = reward + self.gamma * self.rewardCritic(nextState)
        rewardAdvantage = rewardTarget - self.rewardCritic(state)
        costTarget = cost + self.gamma * self.costCritic(nextState)
        costAdvantage = costTarget - self.costCritic(state)

        oldLogProb = []
        for i in range(self.batchSize):
            matchingStateOne = torch.tensor(matchingState[i], dtype=torch.float)
            lPOne = torch.log(torch.softmax(self.actor(matchingStateOne), dim=0)[action[i].item()]).detach()
            oldLogProb.append(lPOne)


        for k in range(self.epochs):
            newLogProb = []
            for i in range(self.batchSize):
                matchingStatetwo = torch.tensor(matchingState[i], dtype=torch.float)
                lP = torch.log(torch.softmax(self.actor(matchingStatetwo), dim=0)[action[i].item()])
                newLogProb.append(lP)
            minRewardList = []
            minCostList = []
            for i in range(self.batchSize):
                ratio = torch.exp(newLogProb[i] - oldLogProb[i])
                rewardSurr1 = ratio * rewardAdvantage[i].detach()
                rewardSurr2 = torch.clamp(ratio, 1 - self.eps,
                                          1 + self.eps) * rewardAdvantage[i].detach()
                costSurr1 = ratio * costAdvantage[i].detach()
                costSurr2 = torch.clamp(ratio, 1 - self.eps,
                                        1 + self.eps) * costAdvantage[i].detach()
                minRewardSurr = torch.min(rewardSurr1, rewardSurr2)
                minCostSurr = torch.min(costSurr1, costSurr2)
                minRewardList.append(minRewardSurr)
                minCostList.append(minCostSurr)
            JR = torch.mean(torch.stack(minRewardList, dim=0), dim=0)
            JC = torch.mean(torch.stack(minCostList, dim=0), dim=0)


            L = JR - self.lagrange * (JC - self.limit)
            lagLoss = L
            lagLoss.backward()
            self.lagrange.data = torch.max(self.lagrange.data -
                                           self.lagLr * self.lagrange.grad,torch.tensor(0,dtype=torch.float))


    def reset_reward_critic_learning_rate(self):
        self.rewardCriticLr = self.rewardCriticLr / 2
        self.rewardCriticOptimizer.param_groups[0]['lr'] = self.rewardCriticLr

    def reset_cost_critic_learning_rate(self):
        self.costCriticLr = self.costCriticLr / 2
        self.costCriticOptimizer.param_groups[0]['lr'] = self.costCriticLr

    def reset_lag_learning_rate(self):
        self.lagLr = self.lagLr / 2

    def reset_actor_learning_rate(self):
        self.actorLr = self.actorLr / 2
        self.actorOptimizer.param_groups[0]['lr'] = self.actorLr