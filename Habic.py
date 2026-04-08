import collections
import random
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim):
        super(PolicyNet, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.S = torch.nn.Linear(self.stateDim, 4)
        self.A = torch.nn.Linear(self.actionDim, 4)
        self.L1 = torch.nn.Linear(4 + 4, 6)
        self.L2 = torch.nn.Linear(6, 4)
        self.f = torch.nn.Linear(4, 1)

    def forward(self, X):
        s = X[:, :self.stateDim]
        a = X[:, -self.actionDim:]
        s1 = F.relu(self.S(s))
        a1 = F.relu(self.A(a))
        y1 = torch.cat((s1, a1), dim=1)
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return self.f(l2)


# --- 创新点 B: Dueling Critic 网络架构 (修复变长候选集问题) ---
class RewardValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(RewardValueNet, self).__init__()
        self.stateDim = stateDim

        # 1. Global Value Stream (V_global: 估算全局价值)
        self.V_S = torch.nn.Linear(stateDim, 16)
        self.V_L1 = torch.nn.Linear(16, 8)
        self.V_L2 = torch.nn.Linear(8, 4)
        self.V_f = torch.nn.Linear(4, 1)

        # 2. Local Advantage Stream (A_local: 估算相对同行的优势)
        self.A_S = torch.nn.Linear(stateDim, 16)
        self.A_L1 = torch.nn.Linear(16, 8)
        self.A_L2 = torch.nn.Linear(8, 4)
        self.A_f = torch.nn.Linear(4, 1)

    def forward(self, X, matchingState_list=None):
        s = X[:, :self.stateDim]

        # 计算全局价值 V
        v_y1 = F.relu(self.V_S(s))
        v_l1 = F.relu(self.V_L1(v_y1))
        v_l2 = F.relu(self.V_L2(v_l1))
        V = self.V_f(v_l2)

        # 计算局部优势 A
        a_y1 = F.relu(self.A_S(s))
        a_l1 = F.relu(self.A_L1(a_y1))
        a_l2 = F.relu(self.A_L2(a_l1))
        A = self.A_f(a_l2)

        if matchingState_list is not None:
            mean_A_list = []
            # 遍历 batch 中的每一个订单，动态处理不同数量的候选人
            for ms_i in matchingState_list:
                cand_states = ms_i[:, :self.stateDim]

                # 计算该订单所有候选人的 Advantage
                cand_a_y1 = F.relu(self.A_S(cand_states))
                cand_a_l1 = F.relu(self.A_L1(cand_a_y1))
                cand_a_l2 = F.relu(self.A_L2(cand_a_l1))
                cand_A = self.A_f(cand_a_l2)  # [N_candidates, 1]

                # 构建 Region Mask：ms_i[:, 0] 是候选司机所在区域
                # ms_i[0, self.stateDim] 是该订单的发起区域
                driver_regions = ms_i[:, 0]
                order_region = ms_i[0, self.stateDim]

                # 1.0 表示同区域候选，0.0 表示跨区域候选
                mask = (driver_regions == order_region).float().unsqueeze(1)

                mask_sum = mask.sum()
                if mask_sum > 0:
                    mean_A_i = (cand_A * mask).sum() / mask_sum
                else:
                    # Fallback 机制：若极其罕见地没有同区域候选人，退化为对所有候选人取均值
                    mean_A_i = cand_A.mean()

                mean_A_list.append(mean_A_i)

            # 将 list 重新组合为 Tensor: [batch_size, 1]
            mean_A_tensor = torch.stack(mean_A_list).unsqueeze(1)

            # 输出 Dueling 融合结果：V + (A - mean(A))
            return V + (A - mean_A_tensor)
        else:
            # nextState 评估时无需相对对比，直接返回基准价值 V
            return V


class CostValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(CostValueNet, self).__init__()
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


class ReplayBuffer:
    def __init__(self, capacity, batchSize):
        self.buffer = collections.deque(maxlen=capacity)
        self.batchSize = batchSize

    def add(self, matchingState, state, action, reward, cost, nextState):
        self.buffer.append((matchingState, state, action, reward, cost, nextState))

    def sample(self):
        transitions = random.sample(self.buffer, self.batchSize)
        matchingState, state, action, reward, cost, nextState = zip(*transitions)
        state = list(state)
        state = [x.tolist() for x in state]
        return matchingState, state, action, reward, cost, nextState

    def size(self):
        return len(self.buffer)


class Habic:
    def __init__(self, stateDim, actionDim, actorLr, criticLr, lagLr, limit, lagrange, epochs, eps, gamma, batchSize):
        self.actor = PolicyNet(stateDim, actionDim)
        self.rewardCritic = RewardValueNet(stateDim)
        self.costCritic = CostValueNet(stateDim)
        self.actorLr = actorLr
        self.rewardCriticLr = criticLr
        self.costCriticLr = criticLr
        self.lagLr = lagLr
        self.limit = limit
        self.gamma = gamma
        self.lagrange = torch.tensor(lagrange, dtype=torch.float, requires_grad=True)
        self.epochs = epochs
        self.eps = eps
        self.batchSize = batchSize
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actorLr)
        self.rewardCriticOptimizer = torch.optim.Adam(self.rewardCritic.parameters(), lr=self.rewardCriticLr)
        self.costCriticOptimizer = torch.optim.Adam(self.costCritic.parameters(), lr=self.costCriticLr)

    def take_action(self, matchingState, dayIndex):
        matchingState = torch.tensor(matchingState, dtype=torch.float)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)
        actionProb = torch.softmax(vOutput, dim=0)
        actionDist = torch.distributions.Categorical(actionProb)
        action = actionDist.sample().cpu()

        return action.item()

    def take_action_test(self, matchingState, dayIndex):
        matchingState = torch.tensor(matchingState, dtype=torch.float)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)
        actionProb = torch.softmax(vOutput, dim=0)
        action = torch.max(actionProb, 0)[1]
        return action.item()

    def update_theta(self, matchingState, state, action, reward, cost, nextState, round, update_k):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action).view(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1)
        cost = torch.tensor(cost, dtype=torch.float).view(-1, 1)
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)

        # 【核心修复】：将 matchingState 转换为 Tensor 的 List，完美兼容长度不一的候选人矩阵
        matchingState_list = [torch.tensor(ms, dtype=torch.float) for ms in matchingState]

        # nextState 只用 V_global 估值，当前 state 用 V+A 融合估值
        nextRewardCritic = self.rewardCritic(nextState)
        rewardCritic = self.rewardCritic(state, matchingState_list)

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

        a, b, c = 0.0, 0.0, 0.0

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
                rewardSurr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * rewardAdvantage[i].detach()
                costSurr1 = ratio * costAdvantage[i].detach()
                costSurr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * costAdvantage[i].detach()

                minRewardSurr = torch.min(rewardSurr1, rewardSurr2)
                minCostSurr = torch.min(costSurr1, costSurr2)
                minRewardList.append(minRewardSurr)
                minCostList.append(minCostSurr)

            JR = torch.mean(torch.stack(minRewardList, dim=0), dim=0)
            JC = torch.mean(torch.stack(minCostList, dim=0), dim=0)

            L = JR - self.lagrange.detach() * JC
            actorLoss = -L

            # 再次评估损失时同样带上 matchingState_list
            rewardCriticLoss = torch.mean(
                F.mse_loss(self.rewardCritic(state, matchingState_list), rewardTarget.detach()))
            costCriticLoss = torch.mean(F.mse_loss(self.costCritic(state), costTarget.detach()))

            if (update_k == 0) & (k == self.epochs - 1):
                a = actorLoss.item()
                b = rewardCriticLoss.item()
                c = costCriticLoss.item()

            self.actorOptimizer.zero_grad()
            self.rewardCriticOptimizer.zero_grad()
            self.costCriticOptimizer.zero_grad()
            actorLoss.backward()
            rewardCriticLoss.backward()
            costCriticLoss.backward()
            self.actorOptimizer.step()
            self.rewardCriticOptimizer.step()
            self.costCriticOptimizer.step()

        return a, b, c

    def update_lagrange(self, matchingState, state, action, reward, cost, nextState, round):
        cost_tensor = torch.tensor(cost, dtype=torch.float)
        expected_cost = torch.mean(cost_tensor).item()

        self.lagrange.data = torch.max(
            self.lagrange.data + self.lagLr * (expected_cost - self.limit),
            torch.tensor(0.0, dtype=torch.float)
        )

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