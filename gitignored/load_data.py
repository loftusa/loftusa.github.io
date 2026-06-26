import torch



def load_data(train_size=50000):

    dataset = torch.load("data/dataset.pt")
    total_size = dataset["matchups"].shape[0]
    torch.manual_seed(0)
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_data = {
        'matchups': dataset["matchups"][train_indices],
        'outcomes': dataset["outcomes"][train_indices],
        'user_ids': dataset["user_ids"][train_indices],
        'prompt_embs': dataset["prompt_embs"][train_indices],
        'response_a_embs': dataset["response_a_embs"][train_indices],
        'response_b_embs': dataset["response_b_embs"][train_indices]
    }
    val_data = {
        'matchups': dataset["matchups"][val_indices],
        'outcomes': dataset["outcomes"][val_indices],
        'user_ids': dataset["user_ids"][val_indices],
        'prompt_embs': dataset["prompt_embs"][val_indices],
        'response_a_embs': dataset["response_a_embs"][val_indices],
        'response_b_embs': dataset["response_b_embs"][val_indices]
    }
    return train_data, val_data, dataset["models"], dataset["users"]