import torch


def gpu_sleep_ms(ms: float, device: torch.device):
    # convert ms -> cycles using device clock rate (approx)
    # On H100, clock ~1.4-1.8 GHz depending on power state; we approximate.
    # Better: calibrate once with events (see Option B below).
    clock_ghz = 1.5
    cycles = int(ms * 1e-3 * clock_ghz * 1e9)
    torch.cuda._sleep(cycles)
