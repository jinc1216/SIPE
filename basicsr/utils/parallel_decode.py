from typing import Callable
import torch
import torch.nn.functional as F
import math


def _step_count(value, name, minimum=0):
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def resolve_num_refine_steps(options):
    """Read the explicit number of refinements after P0."""
    if options.get('num_refine_steps') is None:
        raise ValueError('Specify num_refine_steps explicitly.')
    return _step_count(options['num_refine_steps'], 'num_refine_steps')


def select_predictor_tokens(logits: torch.Tensor, token_selection: str = 'multinomial',
                            temperature: float = 0.5, step: int = 0, num_iter: int = 6):
    """Select every predictor output by argmax or multinomial sampling.

    Temperature is fixed and strictly positive (1.0 gives softmax(logits)).
    Step/horizon arguments are retained for callers but do not alter sampling.
    No token is forbidden and no change is forced. The completion policy
    decides whether to commit all proposals or only masked-position proposals.
    """
    mode = str(token_selection).lower()
    if mode not in ('argmax', 'multinomial'):
        raise ValueError('Evaluator token selection must be argmax or multinomial.')
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError('Sampling temperature must be finite and positive.')
    probs = F.softmax(logits.float() / float(temperature), dim=-1)
    if mode == 'argmax':
        ids = logits.argmax(dim=-1)
    else:
        b, n, k = probs.shape
        ids = torch.multinomial(probs.reshape(b * n, k), 1).reshape(b, n)
    return probs, ids


def resolve_token_update_mode(options):
    """Resolve the token update policy, defaulting to masked positions."""
    mode = str(options.get('token_update_mode', 'masked')).lower()
    if mode not in ('masked', 'full'):
        raise ValueError('token_update_mode must be masked or full.')
    return mode


def predictor_completion(lq_feats, cur_ids, unknown_map, tokens_to_logits,
                          tokens_to_feats, token_selection='multinomial', temperature=0.5,
                          step=0, num_iter=6, logits_lr=None, token_update_mode='masked'):
    """Build masked input context and reconstruct using the chosen update policy.

    In ``full`` mode, every output position comes from the predictor.
    The mask selects LR versus token input features only.
    ``masked`` additionally copies the old output IDs at unmasked positions.
    """
    mode = resolve_token_update_mode({'token_update_mode': token_update_mode})
    b, _, h, w = lq_feats.shape
    safe_ids = torch.where(unknown_map, torch.zeros_like(cur_ids), cur_ids)
    known_feats = tokens_to_feats(safe_ids.reshape(b, 1, h, w)).to(lq_feats)
    mixed_feats = torch.where(unknown_map.reshape(b, 1, h, w), lq_feats, known_feats)
    if logits_lr is not None and bool(unknown_map.all()):
        logits = logits_lr
    else:
        logits = tokens_to_logits(mixed_feats)
    probs, proposed = select_predictor_tokens(
        logits, token_selection, temperature, step=step, num_iter=num_iter)
    completed = proposed if mode == 'full' else torch.where(unknown_map, proposed, cur_ids)
    return logits, probs, completed


def _normalize_mask_len(mask_len: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
    b, n = risk.shape
    if not torch.is_tensor(mask_len):
        mask_len = torch.full((b, 1), int(mask_len), device=risk.device, dtype=torch.long)
    else:
        mask_len = mask_len.to(device=risk.device, dtype=torch.long)
        if mask_len.numel() == 1:
            mask_len = mask_len.reshape(1, 1).expand(b, 1)
        else:
            mask_len = mask_len.reshape(b, 1)
    return mask_len.clamp(0, n)


def exact_topk_mask(mask_len: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
    """Return a deterministic mask with exactly ``mask_len`` highest-risk tokens."""
    _, n = risk.shape
    mask_len = _normalize_mask_len(mask_len, risk)

    order = risk.argsort(dim=-1, descending=True, stable=True)
    take = torch.arange(n, device=risk.device)[None, :] < mask_len
    mask = torch.zeros_like(risk, dtype=torch.bool)
    return mask.scatter(1, order, take)


def resolve_remask_budget_schedule(schedule):
    """Validate the full-grid cosine policy used by X2 decoding."""
    if str(schedule).lower() != 'full_cosine':
        raise ValueError('remask_budget_schedule must be full_cosine.')
    return 'full_cosine'


def _resolve_remask_budget_scale(scale):
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or not 0 < float(scale) <= 1:
        raise ValueError('remask_budget_scale must be finite and in (0, 1].')
    return float(scale)


def cosine_remask_count(num_tokens, state_step, num_refine_steps, budget_scale=1.0):
    """Full-grid budget for action t=state_step+1 in a P0 + T action loop.

    Evaluate the cosine at r=t/(T+1), then use the predictor mixed-input rule:
    floor and clamp to [0, N-1]. A unit scale gives floor(N*cos(pi*r/2));
    optional scaling applies before rounding. N=1 has zero remask positions.
    """
    n = _step_count(num_tokens, 'num_tokens', minimum=1)
    horizon = _step_count(num_refine_steps, 'num_refine_steps', minimum=1)
    step = _step_count(state_step, 'state_step')
    if step >= horizon:
        raise ValueError('A remask decision requires state_step < num_refine_steps.')
    ratio = _resolve_remask_budget_scale(budget_scale) * math.cos(
        math.pi * (step + 1) / (2 * (horizon + 1)))
    return min(n - 1, max(0, math.floor(n * ratio)))


def select_topk_remask(risk, mask_count):
    """Select a fixed full-grid budget, independent of any refinement horizon."""
    if risk.dim() != 2 or risk.shape[1] < 1 or not torch.isfinite(risk).all():
        raise ValueError('Top-K remasking requires finite risk scores of shape (B, N), N >= 1.')
    risk = risk.float()
    b, n = risk.shape
    budget = _normalize_mask_len(mask_count, risk).clamp_max(n - 1)
    # Stable ties select the lowest flattened position index first.
    mask = exact_topk_mask(budget, risk)
    detached = risk.detach()
    last_selected = detached.masked_fill(~mask, torch.inf).amin(-1, keepdim=True)
    first_excluded = detached.masked_fill(mask, -torch.inf).amax(-1, keepdim=True)
    cutoff = torch.where(budget > 0, (last_selected + first_excluded) * 0.5,
                         risk.new_zeros(b, 1))
    return {
        'mask': mask, 'candidate_count': torch.full_like(budget, n),
        'cutoff': cutoff, 'mask_budget': budget,
    }


def correctness_transition_target(old_ids, proposal_ids, gt_ids):
    """Exact label: 0=wrong/wrong, 1=repair, 2=damage, 3=correct/correct."""
    if (old_ids.ndim != 2 or proposal_ids.shape != old_ids.shape
            or gt_ids.shape != old_ids.shape):
        raise ValueError('Correctness transition IDs must share shape (B,N).')
    for ids in (old_ids, proposal_ids, gt_ids):
        if ids.dtype not in (torch.int32, torch.int64) or (ids < 0).any():
            raise ValueError('Transition targets require completed, non-negative integer IDs.')
    return (2 * old_ids.eq(gt_ids).long() + proposal_ids.eq(gt_ids).long()).detach()


def transition_probabilities(logits):
    if logits.ndim != 3 or logits.shape[-1] != 4 or not torch.isfinite(logits).all():
        raise ValueError('Evaluator must return finite (B,N,4) transition logits.')
    return logits.float().softmax(dim=-1)


def transition_error_probability(logits):
    probs = transition_probabilities(logits)
    return probs[..., 0] + probs[..., 1]


def transition_expected_gain(logits):
    probs = transition_probabilities(logits)
    return probs[..., 1] - probs[..., 2]


def accept_token_gains(old_ids, proposal_ids, pair_logits):
    """Maximize predicted expected correct count, with no tuned threshold.

    Each token's GT-ID match is independent of the other accepted IDs, so
    merging selected IDs preserves the count identity. An imperfect learned
    evaluator does not guarantee an increase in the actual correct count.
    """
    gain = transition_expected_gain(pair_logits)
    if gain.shape != old_ids.shape or proposal_ids.shape != old_ids.shape:
        raise ValueError('Transition logits and candidate sequences must align.')
    # Comparing logits avoids losing a tiny probability difference to rounding.
    # softmax(l)[1] > softmax(l)[2] is equivalent to l[1] > l[2].
    accept = (proposal_ids != old_ids) & (pair_logits[..., 1] > pair_logits[..., 2])
    return torch.where(accept, proposal_ids, old_ids), accept, gain


@torch.no_grad()
def propose_evaluator_tokens(lq_feats, old_ids, error_probability, mask_count,
                             tokens_to_logits, tokens_to_feats, *,
                             token_selection='multinomial', sample_temperature=1.0,
                             token_update_mode='full', state_step=0, num_refine_steps=4):
    """The SAME hard action for Stage-3 labels, rollout and deployment."""
    if error_probability.shape != old_ids.shape:
        raise ValueError('Current error probabilities must align with token IDs.')
    decision = select_topk_remask(error_probability.detach(), mask_count)
    mask = decision['mask']
    if not mask.any():
        return old_ids.clone(), decision, 0
    _, _, proposal = predictor_completion(
        lq_feats, old_ids, mask, tokens_to_logits, tokens_to_feats,
        token_selection, sample_temperature, step=state_step + 1,
        num_iter=num_refine_steps + 1, token_update_mode=token_update_mode)
    # Batched full-mode calls must not alter an image with zero requested tokens.
    proposal = torch.where(mask.any(-1, keepdim=True), proposal, old_ids)
    return proposal.detach(), decision, 1


@torch.no_grad()
def decode_evaluator(mask_tokens, lq_feats, tokens_to_logits, tokens_to_feats, *,
                     num_refine_steps, mask_token_id=-1,
                     token_selection='multinomial', sample_temperature=1.0,
                     return_state=False, remask_budget_schedule='full_cosine',
                     token_update_mode='full', remask_budget_scale=1.0,
                     max_refine_steps=None, initial_token_selection=None):
    """Predict -> rank current errors -> propose -> accept positive token gains.

    Evaluator callback receives proposal_ids=... and returns (B,N,4) logits.
    No GT/DINO/image metric participates. After token-wise merging, the next
    step scores the updated state.
    Up to T+1 predictor and 2T evaluator calls; rejected/no-change proposals
    do not terminate the remaining cosine-budget actions.
    """
    horizon = _step_count(num_refine_steps, 'num_refine_steps')
    limit = horizon if max_refine_steps is None else _step_count(max_refine_steps, 'max_refine_steps')
    if limit > horizon:
        raise ValueError('max_refine_steps cannot exceed num_refine_steps.')
    resolve_remask_budget_schedule(remask_budget_schedule)
    scale = _resolve_remask_budget_scale(remask_budget_scale)
    update_mode = resolve_token_update_mode({'token_update_mode': token_update_mode})
    b, _, h, w = lq_feats.shape
    n = h * w
    if tuple(mask_tokens.shape) != (b, n):
        raise ValueError('mask_tokens must have shape (B,H*W).')
    if mask_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError('Initial token IDs must be integers.')
    if ((mask_tokens < 0) & (mask_tokens != mask_token_id)).any():
        raise ValueError('Unexpected negative token ID.')
    initial_selection = token_selection if initial_token_selection is None else initial_token_selection
    initial_mask = mask_tokens.eq(mask_token_id)
    _, _, cur_ids = predictor_completion(
        lq_feats, mask_tokens.long(), initial_mask, tokens_to_logits, tokens_to_feats,
        initial_selection, sample_temperature, 0, horizon + 1,
        token_update_mode=update_mode)
    first_ids = cur_ids.clone()
    seqs = cur_ids[:, None].repeat(1, horizon + 1, 1)
    proposals = seqs.clone()
    masks = torch.zeros_like(seqs, dtype=torch.bool)
    accepts = masks.clone()
    sums = {key: lq_feats.new_zeros(b, dtype=torch.float32) for key in (
        'requested', 'proposal_changed', 'accepted', 'unmasked_accepted',
        'unmasked_positions', 'revisited', 'accepted_gain')}
    previous_mask = torch.zeros_like(cur_ids, dtype=torch.bool)
    predictor_calls, evaluator_calls, steps = 1, 0, 0
    last_decision = None
    for state_step in range(limit):
        count = cosine_remask_count(n, state_step, horizon, scale)
        if count == 0:
            break
        # Self-pair: exactly the same head estimates current correctness.
        self_logits = tokens_to_logits(
            cur_ids, h, w, True, cond_feats=lq_feats, proposal_ids=cur_ids)
        evaluator_calls += 1
        risk = transition_error_probability(self_logits)
        proposal, decision, calls = propose_evaluator_tokens(
            lq_feats, cur_ids, risk, count, tokens_to_logits, tokens_to_feats,
            token_selection=token_selection, sample_temperature=sample_temperature,
            token_update_mode=update_mode, state_step=state_step,
            num_refine_steps=horizon)
        predictor_calls += calls
        pair_logits = tokens_to_logits(
            cur_ids, h, w, True, cond_feats=lq_feats, proposal_ids=proposal)
        evaluator_calls += 1
        updated, accepted, gain = accept_token_gains(cur_ids, proposal, pair_logits)
        mask = decision['mask']
        masks[:, state_step] = mask
        accepts[:, state_step] = accepted
        proposals[:, state_step + 1] = proposal
        last_decision = {'state_step': state_step, 'risk': risk, 'mask': mask,
                         'gain': gain, 'accepted': accepted, 'token_ids': cur_ids}
        sums['requested'] += mask.sum(-1)
        sums['proposal_changed'] += proposal.ne(cur_ids).sum(-1)
        sums['accepted'] += accepted.sum(-1)
        sums['unmasked_accepted'] += (accepted & ~mask).sum(-1)
        sums['unmasked_positions'] += (~mask).sum(-1)
        sums['revisited'] += (mask & previous_mask).sum(-1)
        sums['accepted_gain'] += (gain * accepted).sum(-1)
        cur_ids = updated
        previous_mask = mask
        steps += 1
        seqs[:, steps] = cur_ids
    seqs[:, steps + 1:] = cur_ids[:, None]
    proposals[:, steps + 1:] = cur_ids[:, None]
    denominator = float(max(steps, 1) * n)
    diagnostics = {
        'token_change_rate': cur_ids.ne(first_ids).float().mean(-1),
        'proposal_change_rate': sums['proposal_changed'] / denominator,
        'step_token_change_rate': sums['accepted'] / denominator,
        'accepted_change_rate': sums['accepted'] / sums['proposal_changed'].clamp_min(1),
        'remask_ratio': sums['requested'] / denominator,
        'unmasked_change_rate': sums['unmasked_accepted'] / sums['unmasked_positions'].clamp_min(1),
        'revisited_ratio': sums['revisited'] / denominator,
        'predicted_correct_gain_per_step': sums['accepted_gain'] / max(steps, 1),
        'refine_steps': lq_feats.new_full((b,), float(steps), dtype=torch.float32),
        'predictor_calls': lq_feats.new_full((b,), float(predictor_calls), dtype=torch.float32),
        'evaluator_calls': lq_feats.new_full((b,), float(evaluator_calls), dtype=torch.float32),
    }
    if last_decision is not None:
        diagnostics['last_decision_risk_std'] = last_decision['risk'].std(-1, unbiased=False)
    state = {'state_step': steps, 'num_refine_steps': horizon,
             'token_ids': cur_ids, 'first_ids': first_ids,
             'source_mask': previous_mask, 'last_decision': last_decision,
             'proposal_seqs': proposals, 'accept_history': accepts,
             'diagnostics': diagnostics}
    if return_state:
        return seqs, masks, state
    return seqs, masks
