import torch
import torch.nn as nn
import torch.optim as optim

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels):
        return torch.round(torch.clamp(x, 0, levels - 1))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def quantize_ste(x, levels):
    return QuantizeSTE.apply(x, levels)


class DiscreteNN_STE_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.W1 = nn.Parameter(torch.empty(2, 3))
        self.W2 = nn.Parameter(torch.empty(3, 1))

        nn.init.uniform_(self.W1, -2.0, 2.0)
        nn.init.uniform_(self.W2, -2.0, 2.0)

        self.b1 = nn.Parameter(torch.zeros(3))
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h_raw = (x @ self.W1 + self.b1) * 3.0     # scaling
        h_quant = quantize_ste(h_raw, 4)

        y_raw = (h_quant @ self.W2 + self.b2) * 3.0
        y_quant = quantize_ste(y_raw, 10)

        return y_quant


if __name__ == "__main__":
    net = DiscreteNN_STE_PyTorch()
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    T = torch.tensor([[0.],[1.],[1.],[0.]])

    print("Training...")
    for epoch in range(3000):
        optimizer.zero_grad()
        y_pred = net(X)
        loss = torch.mean((y_pred - T)**2)
        loss.backward()
        optimizer.step()

        if epoch % 300 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.4f}")

    print("\nFinal results:")
    for x,t in zip(X,T):
        y = net(x.unsqueeze(0))
        print(f"{x.numpy()} → predicted={y.detach().numpy()}, target={t.numpy()}")
