import torch
from model.mfanet import CMAD


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    kwargs = dict(
        mode='sinkhorn',
        patch_size=7,
        epsilon=0.05,
        sinkhorn_iter=2,
        balanced_iter=20,
        use_adapter=True,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = torch.randn(1, 256, 25, 25, device=device)
    s = torch.randn(1, 256, 25, 25, device=device)
    merge = torch.randn(1, 256, 25, 25, device=device)

    for mode in ('direct', 'cross_attention', 'pma'):
        model = CMAD(256, pma_kwargs=kwargs, match_mode=mode).to(device).eval()
        with torch.no_grad():
            y = model(q, s, merge, 200, 200)
        print(
            f'{mode:16s} output={tuple(y.shape)} '
            f'finite={bool(torch.isfinite(y).all())} '
            f'CMAD_params={count_params(model)/1e6:.6f}M'
        )


if __name__ == '__main__':
    main()
