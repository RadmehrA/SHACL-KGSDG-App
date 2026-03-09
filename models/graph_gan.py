import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphGenerator(nn.Module):
    def __init__(self, num_subjects, num_predicates, num_objects, hidden_dim=128):
        super(GraphGenerator, self).__init__()
        self.num_subjects = num_subjects
        self.num_predicates = num_predicates
        self.num_objects = num_objects

        
        self.subject_embed = nn.Embedding(num_subjects, hidden_dim)
        self.predicate_embed = nn.Embedding(num_predicates, hidden_dim)

        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_objects)

    def forward(self, s_idx, p_idx):
        s_emb = self.subject_embed(s_idx)
        p_emb = self.predicate_embed(p_idx)
        x = torch.cat([s_emb, p_emb], dim=-1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class GraphDiscriminator(nn.Module):
    def __init__(self, num_subjects, num_predicates, num_objects, hidden_dim=128):
        super(GraphDiscriminator, self).__init__()
        self.subject_embed = nn.Embedding(num_subjects, hidden_dim)
        self.predicate_embed = nn.Embedding(num_predicates, hidden_dim)
        self.object_embed = nn.Embedding(num_objects, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s_idx, p_idx, o_idx):
        s_emb = self.subject_embed(s_idx)
        p_emb = self.predicate_embed(p_idx)
        o_emb = self.object_embed(o_idx)
        x = torch.cat([s_emb, p_emb, o_emb], dim=-1)
        x = F.relu(self.fc1(x))
        logit = self.fc2(x)
        return torch.sigmoid(logit)