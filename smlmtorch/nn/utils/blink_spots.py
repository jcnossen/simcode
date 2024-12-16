import torch

def sample_spot_frames(batch_size, k_on, k_off, max_frames):
    """
    Sample t_start and t_end for a Markov process with k_on and k_off in steady state.

    Args:
        batch_size (int): Number of samples to generate.
        k_on (float): On rate per frame.
        k_off (float): Off rate per frame.
        max_frames (int): Maximum number of frames for each process.

    Returns:
        t_start (Tensor): Starting frames of shape (batch_size,).
        t_end (Tensor): Ending frames of shape (batch_size,).
    """

    p_on = k_on / (k_on + k_off)
    t_start = torch.zeros(batch_size, dtype=torch.long)
    t_end = torch.zeros(batch_size, dtype=torch.long)

    # Sample whether each process is on or off at t = 0
    on_at_t0 = torch.rand(batch_size) < p_on

    # Sample t_start from a geometric distribution with probability k_on
    t_start_distr = torch.distributions.geometric.Geometric(k_on)
    t_start[~on_at_t0] = t_start_distr.sample((batch_size - on_at_t0.sum(),)).long()

    # Clamp t_start to the range [0, max_frames]
    t_start = torch.clamp(t_start, 0, max_frames)

    # Sample t_end from a geometric distribution with probability k_off
    t_end_offset = torch.distributions.geometric.Geometric(k_off).sample((batch_size,)).long()

    # Calculate t_end as t_start + t_end_offset + 1
    t_end = t_start + t_end_offset + 1

    # Clamp t_end to the range [t_start + 1, max_frames]
    t_end = torch.min(t_end, torch.full_like(t_end, max_frames))

    return t_start, t_end

if __name__ == "__main__":

    # Example usage:
    batch_size = 10
    k_on = 0.1
    k_off = 0.1
    max_frames = 30

    t_start, t_end = sample_spot_frames(batch_size, k_on, k_off, max_frames)
    print("t_start:", t_start)
    print("t_end:", t_end)
