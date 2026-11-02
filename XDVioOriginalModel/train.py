import torch


def CLAS(logits, label, seq_len, criterion, device, is_topk=True):
    del device
    logits = logits.squeeze(-1)
    seq_len = seq_len.to(logits.device)
    positions = torch.arange(logits.shape[1], device=logits.device)
    valid = positions.unsqueeze(0) < seq_len.unsqueeze(1)

    if is_topk:
        # Calculate enough candidates for the longest possible sequence, then
        # use a rank mask to retain each sample's original variable k.
        max_k = logits.shape[1] // 16 + 1
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        top_values = torch.topk(masked_logits, k=max_k, dim=1, largest=True).values
        sample_k = (seq_len // 16 + 1).clamp(max=max_k)
        ranks = torch.arange(max_k, device=logits.device).unsqueeze(0)
        topk_mask = ranks < sample_k.unsqueeze(1)
        instance_logits = (top_values * topk_mask).sum(1) / sample_k
    else:
        instance_logits = (logits * valid).sum(1) / seq_len.clamp_min(1)

    instance_logits = torch.sigmoid(instance_logits)

    clsloss = criterion(instance_logits, label)
    return clsloss


def CENTROPY(logits, logits2, seq_len, device):
    del device
    logits = logits.squeeze(-1)
    logits2 = logits2.squeeze(-1)
    seq_len = seq_len.to(logits.device)
    positions = torch.arange(logits.shape[1], device=logits.device)
    valid = positions.unsqueeze(0) < seq_len.unsqueeze(1)
    probabilities = torch.sigmoid(logits)
    auxiliary_probabilities = torch.sigmoid(logits2)
    losses = -probabilities.detach() * torch.log(auxiliary_probabilities)
    return (losses * valid).sum(1).div(seq_len.clamp_min(1)).mean()


def train(dataloader, model, optimizer, criterion, device, is_topk, profiler=None):
    running_losses = torch.zeros(4, device=device)
    batches = 0

    with torch.set_grad_enabled(True):
        model.train()
        for i, (input, label) in enumerate(dataloader):
            with torch.profiler.record_function('host_to_device'):
                input = input.float().to(device, non_blocking=True)
                label = label.float().to(device, non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0] > 0, 1)

            with torch.profiler.record_function('forward_backward_optimizer'):
                logits, logits2 = model(input, seq_len)
                clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
                clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
                croloss = CENTROPY(logits, logits2, seq_len, device)

                total_loss = clsloss + clsloss2 + 5*croloss
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            running_losses += torch.stack((clsloss.detach(), clsloss2.detach(),
                                           croloss.detach(), total_loss.detach()))
            batches += 1
            if profiler is not None:
                profiler.step()

    if batches == 0:
        raise RuntimeError('The training data loader produced no batches')

    average_losses = (running_losses / batches).cpu().tolist()
    return dict(zip((
        'train/classification_loss',
        'train/aux_classification_loss',
        'train/cross_entropy_loss',
        'train/total_loss',
    ), average_losses))
