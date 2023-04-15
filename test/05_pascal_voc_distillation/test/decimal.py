import torch

class RoundDecimal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_digits):
        ctx.save_for_backward(input)
        ctx.n_digits = n_digits
        return torch.round(input * 10**n_digits) / (10**n_digits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return torch.round(grad_input * 10**ctx.n_digits) / (10**ctx.n_digits), None


torch_round_x_decimal = RoundDecimal.apply

def test_round_decimal():

    x = torch.Tensor([1.234, 4.567, 6.789])
    x = x.detach()
    x.requires_grad = True

    # y = torch_round_x_decimal(x, 2)
    y = x
    y = torch.sum(torch.Tensor([3.141523]) * y)
    y.backward()

    print(x.grad)
    assert torch.all(x.grad == 3.1415).item()

    x = torch.Tensor([1.234, 4.567, 6.789])
    x = x.detach()
    x.requires_grad = True

    # y = torch_round_x_decimal(x, 3)
    y = x
    y = torch.sum(torch.Tensor([3.141523]) * y)
    y.backward()

    print(x.grad)
    assert torch.all(x.grad == 3.1415).item()

if __name__ == "__main__":
    test_round_decimal()