def update_target_network(target_params, source_params, tau=1.0):
    for t, s in zip(target_params, source_params):
        t.data.copy_(tau * s.data + (1.0 - tau) * t.data)