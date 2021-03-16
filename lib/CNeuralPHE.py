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
    return device


class MLP_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(MLP_model, self).__init__()
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


class obs_data(torch.utils.data.Dataset):
    def __init__(self, alpha):
        self.user_history = []
        self.context_history = []
        self.click_history = []
        self.size = 65536 * 16
        self.grid = 64
        self.alpha = alpha

    def push(self, userid, context, click):
        self.user_history.append(userid)
        self.context_history.append(context)
        self.click_history.append(click)
        if len(self.context_history) >= self.size:
            self.user_history = self.user_history[self.grid :]
            self.context_history = self.context_history[self.grid :]
            self.click_history = self.click_history[self.grid :]
            print(len(self.user_history))

    def __len__(self):
        return len(self.user_history)

    def __getitem__(self, idx):
        return {
            "user": self.user_history[idx],
            "context": torch.from_numpy(self.context_history[idx]).to(torch.float),
            "click": torch.tensor(
                self.click_history[idx]
                + np.random.normal(
                    0,
                    self.alpha,
                ),
                dtype=torch.float,
            ),
        }


class CNeuralPHEAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        # set parameters
        self.device = get_device()
        self.args = arg_dict
        for key in arg_dict:
            setattr(self, key, arg_dict[key])

        self.learner = MLP_model(input_dim=self.dimension, hidden_layers=self.hidden_layers).to(self.device)
        self.loss_func = torch.nn.MSELoss().to(self.device)
        # load user features from the designated path based on the dataset.
        self.user_feature = np.genfromtxt(self.path, delimiter=" ")

        self.data = obs_data(self.alpha)
        self.cnt = 0
        self.batch = 1

        # self.A = (self.lambda_ * torch.ones(self.learner.total_param)).to(self.device)
        # self.g = None

    def decide(self, pool_articles, userID, k=1):
        n_article = len(pool_articles)
        # user_vec = torch.cat(n_articles * [self.user_feature[userID - 1].view(1, -1)])
        # article_vec = torch.cat([torch.])
        user_vec = self.user_feature[userID - 1].reshape(1, -1)
        feature_tensor = torch.cat(
            [
                torch.from_numpy(
                    np.concatenate((user_vec, x.contextFeatureVector[: self.dimension].reshape(1, -1)), 1)
                ).to(torch.float32)
                for x in pool_articles
            ]
        ).to(self.device)
        score = self.learner(feature_tensor).view(-1)
        # sum_score = torch.sum(score)
        # with backpack(BatchGrad()):
        #     sum_score.backward()

        # grad = torch.cat([p.grad_batch.view(n_article, -1) for p in self.learner.parameters()], dim=1)
        # uncertainty = torch.sqrt(torch.sum(grad * grad / self.A, dim=1))
        arm = torch.argmax(score).item()
        # self.g = grad[arm]
        return [pool_articles[arm]]

    def updateParameters(self, articlePicked, click, userID):
        user_vec = self.user_feature[userID - 1].reshape(1, -1)
        article_vec = articlePicked.contextFeatureVector[: self.dimension].reshape(1, -1)
        context_vec = np.concatenate((user_vec, article_vec), 1)
        # self.A += self.g * self.g
        self.data.push(userID, context_vec, click)
        self.cnt = (self.cnt + 1) % self.batch

        # batch update, if self.batch == 1, then update in every round
        if self.cnt % self.batch == 0:
            optimizer = torch.optim.Adam(
                self.learner.parameters(), lr=self.learning_rate, weight_decay=self.lambda_ / len(self.data)
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.learning_rate_decay)
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=1024, shuffle=True, num_workers=0)
            loss_list = []
            # early_cnt = 0
            if len(self.data) < 1024:
                num_batch = 1
            else:
                num_batch = int(len(self.data) / 1024)

            for i in range(100):
                total_loss = 0
                for j, batch in enumerate(dataloader):
                    if j == num_batch:
                        break
                    self.learner.zero_grad()
                    optimizer.zero_grad()
                    context_feature = batch["context"].to(self.device)
                    clicks = batch["click"].to(self.device)
                    pred = self.learner(context_feature).view(-1)
                    loss = self.loss_func(pred, clicks)
                    total_loss += loss
                    loss.backward()
                    optimizer.step()

                loss_list.append((total_loss).item() / num_batch)
                # if i != 0 and loss_list[-1] < total_loss / (j + 1):
                #     early_cnt += 1
                #     if early_cnt == 5:
                #         break
                # else:
                #     early_cnt = 0

                # early stop
                if total_loss / num_batch < 1e-2:
                    break

                if i > 20:
                    mean = sum(loss_list[-5:]) / 5
                    mini = min(loss_list[-5:])
                    if (mean / mini - 1) < 0.05:
                        print("No much change {}".format(loss_list[-5:]))
                        break

                scheduler.step()

            if len(self.data) % 100 == 0:
                print("Round: {} learning_rate {}".format(len(self.data), self.learning_rate))
                print([loss for loss in loss_list])
            print("samples {: >5} iters {: >5} final loss {}".format(len(self.data), i, loss_list[-1]))