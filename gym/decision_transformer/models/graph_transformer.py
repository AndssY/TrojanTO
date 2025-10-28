import torch
from torch import nn, einsum
from einops import rearrange, repeat

from decision_transformer.models.GT import GraphTransformer as GT
from decision_transformer.models.GT import GraphTransformerPlus as GTP

class GraphTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        model_type='gtn',
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        depth=6,
        heads = 8,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.model_type = model_type

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_return = torch.nn.Linear(1, hidden_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_dim)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_dim)
        self.embed_edges = nn.Embedding(2, hidden_dim)

        self.embed_ln = nn.LayerNorm(hidden_dim)

        self.predict_state = torch.nn.Linear(hidden_dim, self.state_dim)

        self.embed_ln2 = nn.LayerNorm(2 * hidden_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(2 * hidden_dim, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_return = torch.nn.Linear(hidden_dim, 1)

        if 'GTP' in model_type:
            self.layers = GTP(
                depths=depth,
                embed_dim=hidden_dim,
                ff_embed_dim=hidden_dim * 2,
                num_heads=heads,
                dropout=0.1,
                model_type=model_type,
                state_dim=state_dim,
                max_T=max_length
            )
        else:
            self.layers = GT(
                depths=depth,
                embed_dim=hidden_dim,
                ff_embed_dim=hidden_dim * 2,
                num_heads=heads,
                dropout=0.1,
                model_type=model_type
            )

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):
        # s_0, a_0, R_0, s_1, a_1, R_1
        # to construct the graph
        batch_size, seq_length = states.shape[0], states.shape[1]
        device = states.device

        mask = torch.tril(torch.ones(seq_length * 3, seq_length * 3)).to(device=device) == 0

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        if 'rwd' in self.model_type:
            returns_embeddings = self.embed_return(rewards)
        else:
            returns_embeddings = self.embed_return(returns_to_go)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # nodes (batch_size, seq_length * 3, hidden_size)
        token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 3 , self.hidden_dim), dtype=torch.float32, device=device)
        if "rwd" in self.model_type:
            token_embeddings[:, ::3, :] = state_embeddings
            token_embeddings[:, 1::3, :] = action_embeddings
            token_embeddings[:, 2::3, :] = returns_embeddings
        else:
            token_embeddings[:, ::3, :] = returns_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        nodes = token_embeddings

        # edges (batch_size,  seq_length * 3, seq_length * 3, 1)
        if "rwd" in self.model_type:
            edges = torch.zeros(1, seq_length * 3, seq_length * 3)
            for j in range(seq_length):
                b_index = j * 3
                edges[0, b_index, b_index + 1] = 1
                edges[0, b_index, b_index + 2] = 1
                edges[0, b_index + 1, b_index + 2] = 1
                if b_index + 3 < seq_length * 3:
                    edges[0, b_index, b_index + 3] = 1
                    edges[0, b_index + 1, b_index + 3] = 1
        else:
            edges = torch.zeros(1, seq_length * 3, seq_length * 3)
            for j in range(seq_length):
                b_index = j * 3
                edges[0, b_index, b_index + 2] = 1
                edges[0, b_index + 1, b_index + 2] = 1
                if b_index + 3 < seq_length * 3:
                    edges[0, b_index, b_index + 3] = 1
                    edges[0, b_index + 1, b_index + 3] = 1
                    edges[0, b_index + 1, b_index + 4] = 1
                    edges[0, b_index + 2, b_index + 3] = 1
                    edges[0, b_index + 2, b_index + 4] = 1
        edges = edges.transpose(1, 2)
        edges = edges.repeat(batch_size, 1, 1).to(device=device)
        edges = self.embed_edges(edges.type(torch.long))

        nodes = nodes.permute(1, 0, 2).contiguous() # seq_length x bs x dim
        edges = edges.permute(1, 2, 0, 3).contiguous() # seq_length x seq_length x bs x dim
        if 'GTP' in self.model_type:
            nodes, out = self.layers(nodes, edges, states, self_attn_mask=mask)
        else:
            nodes, out = self.layers(nodes, edges, self_attn_mask=mask)
        nodes = nodes.permute(1, 0, 2).contiguous() # bs x seq_length  x dim


        nodes = nodes.reshape(batch_size, seq_length, 3, -1).permute(0, 2, 1, 3)
        logist = self.embed_ln2(out)
        actions_preds = self.predict_action(logist)

        state_preds = self.predict_state(nodes[:, 2])
        returns_preds = self.predict_return(nodes[:, 2])

        return state_preds, actions_preds, returns_preds

    def get_action(self, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        if rewards is not None:
            rewards = rewards.reshape(1, -1, 1)
        if returns_to_go is not None:
            returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            if rewards is not None:
                rewards = rewards[:, -self.max_length:]
            if returns_to_go is not None:
                returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # padding
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            if rewards is not None:
                rewards = torch.cat(
                    [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                    dim=1).to(dtype=torch.float32)
            if returns_to_go is not None:
                returns_to_go = torch.cat(
                    [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                    dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(states, actions, rewards, None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]

if __name__ == "__main__":
    model = GraphTransformer(
        state_dim=16,
        action_dim=3,
        hidden_dim=8,
        max_length=2,
    )
    states = torch.randn(2,2, 16)
    actions = torch.randn(2,2,3)
    rewards = torch.randn(2,2,1)
    returns_to_go = torch.randn(2,2,1)
    timesteps = torch.tensor([[0,1],[2,3]]).to(dtype=torch.long)
    model(states=states, actions=actions,rewards=rewards, targets=None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=None)