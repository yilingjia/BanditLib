import numpy as np
import torch
from lib.BaseAlg import BaseAlg
from backpack import backpack, extend
from backpack.extensions import BatchGrad


def get_device():
    if torch.cuda.is_available():
        device = "cuda:{}".format(torch.cuda.current_device())
    else:
        device = "cpu"
    # print("Use device: ", device)
    return device
    # return "cpu"


class MLP_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(MLP_model, self).__init__()
        # self.learning_rate = learning_rate
        # self.learning_rate_decay = learning_rate_decay
        # self.threshold = threshold
        # self.iteration = iteration
        # self._lambda = _lambda
        # self.device = device

        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_layers[0]))
        layers.append(torch.nn.ReLU())
        for idx in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[idx], hidden_layers[idx + 1]))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_layers[-1], 1))
        self.model = torch.nn.Sequential(*layers)
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.loss_func = torch.nn.MSELoss()
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, input_):
        out = self.model(input_)
        return out

    # def update_model(self, context, label):

    #     n_sample = len(context)
    #     self.optim = torch.optim.Adam(
    #         self.model.parameters(), lr=self.learning_rate, weight_decay=self._lambda / n_sample
    #     )
    #     context_tensor = torch.tensor(context, dtype=torch.float32).to(self.device)
    #     label_tensor = torch.tensor(label, dtype=torch.float32).to(self.device).view(-1)

    #     for _ in range(100):
    #         self.model.zero_grad()
    #         self.optim.zero_grad()
    #         pred = self.model(context_tensor).view(-1)
    #         loss = self.loss_func(pred, label_tensor)
    #         loss.backward()
    #         self.optim.step()
    #     return 0


class NeuralUserStruct:
    def __init__(self, featureDimension, hidden_layers, lambda_, alpha, learning_rate, learning_rate_decay):
        self.device = get_device()
        self.learner = extend(MLP_model(featureDimension, hidden_layers).to(self.device))
        self.loss_func = extend(torch.nn.MSELoss().to(self.device))
        self.context_history = []
        self.click_history = []
        self.lambda_ = lambda_
        self.alpha = alpha
        self.A = (self.lambda_ * torch.ones(self.learner.total_param)).to(self.device)
        self.g = None
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        # .to(self.device)

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.context_history.append(articlePicked_FeatureVector)
        self.click_history.append(click)
        n_sample = len(self.context_history)
        self.optim = torch.optim.Adam(
            self.learner.parameters(), lr=self.learning_rate, weight_decay=self.lambda_ / n_sample
        )
        scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=self.learning_rate_decay)
        self.A += self.g * self.g
        context_tensor = torch.tensor(self.context_history, dtype=torch.float32).to(self.device)
        label_tensor = torch.tensor(self.click_history, dtype=torch.float32).to(self.device).view(-1)
        # self.learner.to(self.device)
        # self.loss_func.to(self.device)

        for i in range(100):
            self.learner.zero_grad()
            self.optim.zero_grad()
            pred = self.learner(context_tensor).view(-1)
            loss = self.loss_func(pred, label_tensor)
            loss.backward()
            self.optim.step()
            scheduler.step()
            # prev_loss = loss

            # early stop
            if loss < 1e-3:
                break
            # if i != 0 and prev_loss < loss:
            #     early_cnt += 1
            # else:
            #     early_cnt = 0
            # if early_cnt >= 5:
            #     break
        print("samples {: >5} iters {: >5} final loss {}".format(len(self.context_history), i, loss))


class NeuralUCBAlgorithm(BaseAlg):
    def __init__(self, arg_dict, init="random"):  # n is number of users
        BaseAlg.__init__(self, arg_dict)
        self.args = arg_dict
        self.users = []
        self.device = get_device()
        # algorithm have n users, each user has a user structure
        for _ in range(arg_dict["n_users"]):
            self.users.append(
                NeuralUserStruct(
                    arg_dict["dimension"],
                    arg_dict["hidden_layers"],
                    arg_dict["lambda_"],
                    arg_dict["alpha"],
                    arg_dict["learning_rate"],
                    arg_dict["learning_rate_decay"],
                )
            )

    def decide(self, pool_articles, userID, k=1):
        # construct tensor feature of the pool articles
        n_article = len(pool_articles)
        article_tensor = torch.cat(
            [
                torch.from_numpy(x.contextFeatureVector[: self.dimension]).view(1, -1).to(torch.float32)
                for x in pool_articles
            ]
        ).to(self.device)
        score = self.users[userID].learner(article_tensor).view(-1)
        sum_score = torch.sum(score)
        with backpack(BatchGrad()):
            sum_score.backward()

        grad = torch.cat([p.grad_batch.view(n_article, -1) for p in self.users[userID].learner.parameters()], dim=1)
        uncertainty = torch.sqrt(torch.sum(grad * grad / self.users[userID].A, dim=1))
        arm = torch.argmax(score + self.alpha * uncertainty).item()
        # print(
        #     "user {}, history {}, selected arm: score {} uncertainty {} CB {}".format(
        #         userID,
        #         self.users[userID].click_history,
        #         score[arm],
        #         uncertainty[arm],
        #         self.alpha * uncertainty[arm],
        #     )
        # )
        self.users[userID].g = grad[arm]
        return [pool_articles[arm]]

    def getProb(self, pool_articles, userID):
        means = []
        vars = []
        for x in pool_articles:
            x_pta, mean, var = self.users[userID].getProb_plot(self.alpha, x.contextFeatureVector[: self.dimension])
            means.append(mean)
            vars.append(var)
        return means, vars

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.contextFeatureVector[: self.dimension], click)

    ##### SHOULD THIS BE CALLED GET COTHETA #####
    def getCoTheta(self, userID):
        return self.users[userID].UserTheta

    def getTheta(self, userID):
        return self.users[userID].UserTheta
