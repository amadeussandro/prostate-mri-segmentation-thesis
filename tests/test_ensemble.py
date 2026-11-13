import torch
import torch.nn as nn

import os

from src.ensemble import LEFT_RIGHT_DIM, SoftmaxEnsemble, resolve_member_path


def test_member_path_written_on_windows_resolves_anywhere():
    got = resolve_member_path("/content/bundle", "results/exp_e1_dice_ce\\best_model.pt")
    assert got == os.path.join("/content/bundle", "results", "exp_e1_dice_ce", "best_model.pt")
    assert "\\" not in got.replace(os.sep, "/") or os.sep == "\\"
    assert resolve_member_path("/r", "results/a/b.pt") == os.path.join("/r", "results", "a", "b.pt")


class _Lin(nn.Module):
    """1x1 conv: per-pixel logits, so flips commute with it."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(1, 3, 1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(3, 1, 1, 1, generator=g))
            self.conv.bias.copy_(torch.randn(3, generator=g))

    def forward(self, x):
        return self.conv(x)


class _Pos(nn.Module):
    """Logits depend on row position, so a flip changes the prediction."""

    def forward(self, x):
        rows = torch.arange(x.shape[2], dtype=x.dtype).view(1, 1, -1, 1).expand_as(x)
        return torch.cat([rows, -rows, torch.zeros_like(rows)], dim=1)


def test_single_member_matches_model_argmax():
    m = _Lin(0).eval()
    x = torch.randn(2, 1, 8, 6)
    ens = SoftmaxEnsemble([m]).eval()
    assert torch.equal(ens(x).argmax(1), m(x).argmax(1))
    assert torch.allclose(ens(x).sum(1), torch.ones(2, 8, 6))


def test_average_of_softmax():
    a, b = _Lin(1).eval(), _Lin(2).eval()
    x = torch.randn(1, 1, 5, 5)
    expected = (a(x).softmax(1) + b(x).softmax(1)) / 2
    assert torch.allclose(SoftmaxEnsemble([a, b])(x), expected)


def test_flip_tta_is_undone_and_on_left_right_axis():
    m = _Lin(3).eval()
    x = torch.randn(1, 1, 7, 4)
    # A per-pixel model is flip-equivariant, so TTA must give the same map.
    assert torch.allclose(SoftmaxEnsemble([m], tta_flip=True)(x), m(x).softmax(1), atol=1e-6)
    # A position-dependent model shows the flip happens on dim 2 (left-right).
    p = _Pos()
    out = SoftmaxEnsemble([p], tta_flip=True)(x)
    flipped = torch.flip(p(torch.flip(x, dims=[LEFT_RIGHT_DIM])).softmax(1), dims=[LEFT_RIGHT_DIM])
    assert torch.allclose(out, (p(x).softmax(1) + flipped) / 2)
    assert LEFT_RIGHT_DIM == 2
