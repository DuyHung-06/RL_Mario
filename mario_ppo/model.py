
"""Mạng actor-critic cho ảnh Mario.

Giao diện mà train.py và evaluate.py dùng (giữ nguyên tên và chữ ký):
  ActorCritic(n_actions, in_channels=4)
  forward(obs)  : obs uint8 (B, C, 84, 84) -> (logits (B, n_actions) float32, value (B,) float32)
  act(obs)      : lấy mẫu hành động, trả (action long (B,), logp (B,), value (B,)) trên CPU, không gradient
"""
import numpy as np
import torch
from torch import nn


def _init(layer, gain=np.sqrt(2)):
    """Khởi tạo trực giao, bias bằng 0 (thường dùng cho PPO)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Thân CNN dùng chung (kiểu Nature DQN), hai đầu ra: logits của chính sách và giá trị trạng thái.

    Mục đích: một mạng vừa là actor (chọn hành động) vừa là critic (ước lượng giá trị cho GAE).
    Kích thước: 84x84 -> conv8/4 -> 20x20 -> conv4/2 -> 9x9 -> conv3/1 -> 7x7 (64 kênh) -> 3136 -> hidden.
    """

    def __init__(self, n_actions, in_channels=4, hidden=512):
        super().__init__()
        self.body = nn.Sequential(
            _init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(64 * 7 * 7, hidden)), nn.ReLU(),
        )
        # Đầu chính sách: gain nhỏ để chính sách ban đầu gần đều
        self.policy = _init(nn.Linear(hidden, n_actions), gain=0.01)
        # Đầu giá trị: 1 số
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs):
        x = obs.float() / 255.0  # uint8 [0, 255] -> float [0, 1]
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """CÓ SẴN. Lấy mẫu từ chính sách hiện tại; nhận obs ở thiết bị bất kỳ, trả tensor CPU."""
        device = next(self.parameters()).device
        logits, value = self(obs.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu(), value.cpu()
