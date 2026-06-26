import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from load_data import load_data

# Task 1
def compute_loss(probs, outcomes):
    """Loss function to optimize and re-use across model variants"""
    # TODO
    pass

class BradleyTerryModel(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.ratings = nn.Parameter(torch.randn(num_models))
    
    def forward(self, data):
        # TODO
        probs = None
        return probs

# Task 2
class UserAwareBTModel(nn.Module):
    def __init__(self, num_models, num_users):
        """Feel free to add more arguments as needed."""
        super().__init__()
        self.ratings = nn.Parameter(torch.randn(num_models))
        # TODO
    
    def forward(self, data):
        # TODO
        probs = None
        return probs

# Task 3
class EmbeddingBTModel(nn.Module):
    def __init__(self, num_models, num_users, content_emb_dim):
        """Feel free to add more arguments as needed."""
        super().__init__()
        self.ratings = nn.Parameter(torch.randn(num_models))
        # TODO
    
    def forward(self, data):
        # TODO
        probs = None
        return probs


# Boiler plate training and evaluation code
def train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    probs = model(data)
    loss = compute_loss(probs, data['outcomes'])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        probs = model(data)
        outcomes = data['outcomes']
        loss = compute_loss(probs, outcomes)
        predictions = (probs > 0.5).float()
        correct = (predictions == (outcomes > 0.5).float()).float()
        accuracy = correct.mean()
    return loss.item(), accuracy.item()

def train_loop(model, optimizer, train_data, val_data, num_epochs=200):
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, train_data)
        if (epoch + 1) % 10 == 0:
            val_loss, val_acc = evaluate(model, val_data)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


def print_leaderboard(ratings, models, top_k=20):
    print(f"Top {top_k} Model ratings (higher is better):")
    ratings = ratings.detach().numpy()
    model_rankings = [(name, strength) for name, strength in zip(models, ratings)]
    model_rankings.sort(key=lambda x: x[1], reverse=True)
    for rank, (model_name, strength) in enumerate(model_rankings[:top_k], 1):
        print(f"{rank:2d}. {model_name}: {strength:.3f}")


def main():
    train_data, val_data, model_names, users = load_data(train_size=50000)
    print(f"Train size: {train_data['matchups'].shape[0]}")
    print(f"Val size: {val_data['matchups'].shape[0]}")
    print(f"Number of models: {len(model_names)}")
    print(f"Number of users: {len(users)}")
    num_models = len(model_names)
    num_users = len(users)

    # Task 1
    model = BradleyTerryModel(num_models)

    # Task 2
    # model = UserAwareBTModel(num_models, num_users=num_users)

    # Task 3
    # model = EmbeddingBTModel(num_models, num_users=num_users, content_emb_dim=32)

    # Feel free to change these
    lr = 0.02
    optimizer = Adam(model.parameters(), lr=lr)

    train_loop(model, optimizer, train_data, val_data, num_epochs=200)
    print_leaderboard(model.ratings, model_names)


if __name__ == "__main__":
    main()