# Human-Power Assistant for MBAG: Design

Date: 2026-09-11
Branch family: `human-power/*`
Status: approved in discussion, implementation starting

## 1. Goal

Replace the AssistanceZero objective (shared goal-progress reward plus goal
inference) with the intrinsic **long-term human power** objective from

> Heitzig, J. and Potham, R. (2025). *Model-Based Soft Maximization of
> Suitable Metrics of Long-Term Human Power.* arXiv:2508.00159v2.

The assistant ("robot" $r$) must never receive goal-progress reward and must
never be told the human's goal. Its reward is the paper's $U_r(s')$, computed
from an estimate of the human's **ICCEA power** $W_h(s')$.

The scientific question is whether an assistant that only maximises human
power still helps people build houses, measured with MBAG's existing
`goal_percentage` metric and the number of human actions to completion.

Compute target for the first iteration is CPU only (Apple Silicon laptop).
Configs must be written so that the paper-scale settings are a flag flip.

## 2. The equations being implemented

Paper notation, fully observed case (Table 1 of the paper). One human, so
the sum over $h$ has one term.

| Eq. | Definition | Role |
|-----|------------|------|
| (2) | $\pi_h(s,g) = \nu\,\pi^0_h(s,g) + (1-\nu)\,\mathrm{softmax}_{\beta_h} Q^m_h(s,g,\cdot)$ | human behaviour prior |
| (6) | $V^e_h(s,g) = \mathbb E\big[U_h(s',g) + \gamma_h V^e_h(s',g)\big]$, $U_h(s',g)=\mathbb 1[s'\in g]$ | discounted goal-attainment probability under actual $\pi_r,\pi_h$ |
| (7) | $X_h(s) = \sum_{g\in\mathcal G_h} V^e_h(s,g)^{\zeta}$ | effective number of attainable goals |
| (10) | $W_h(s) = \log_2 X_h(s)$ | ICCEA power in bits |
| (8) | $U_r(s) = -\big(\sum_h X_h(s)^{-\xi}\big)^{\eta}$ | robot intrinsic reward for being in $s$ |
| (4) | $Q_r(s,a_r) = \mathbb E\,\gamma_r V_r(s')$ | |
| (9) | $V_r(s) = U_r(s) + \mathbb E_{a_r\sim\pi_r} Q_r(s,a_r)$ | |
| (5) | $\pi_r(s)(a) \propto (-Q_r(s,a))^{-\beta_r}$ | soft power-law policy |

Learning targets for Phase 2 (Section 3.1 of the paper):

| Eq. | Target |
|-----|--------|
| (12) | $v_r(s) \leftarrow \gamma_r V_r(s')$ |
| (13) | $v^e_h(s,g) \leftarrow U_h(s',g) + \gamma_h V^e_h(s',g)$ |
| (14) | $x_h(s) \leftarrow |\mathcal G_h|\, V^e_h(s,g)^{\zeta}$, with $g$ the goal active in the sampled transition |

Paper hyperparameters (Tables 3 to 5): $\zeta=2$, $\xi=1$, $\eta=1.1$,
$\gamma_h=\gamma_r=0.99$, $\beta_r$ annealed $1\to5$, goal-change probability
$p_g=0.01$ per step.

Appendix F.3 states the per-transition robot reward is $U_r(s')$, the value
of the **next** state.

## 3. Mapping onto MBAG

| Paper object | MBAG realisation |
|---|---|
| robot $r$ | assistant, player index 1, policy id `assistant` |
| human $h$ | player index 0, policy id `human` |
| state $s$ | `MbagEnv` world state: blocks, inventories, positions |
| goal set $\mathcal G_h$ | the goal generator's distribution of houses. A goal $g$ is the event "world blocks equal blueprint $g$". Completion is terminal when `terminate_on_goal_completion=True`, which gives the paper's mutual-unreachability property |
| $U_h(s',g)$ | `1` on the step in which `goal_percentage` reaches `1.0`, else `0`. Computed from env infos, never from goal-progress reward |
| $\pi_h(s,g)$ | a frozen goal-conditioned human policy. Two choices: (a) a heuristic goal-following agent (`LayerBuilderAgent`) for CPU smoke tests; (b) the piKL `human` policy stored in `data/assistancezero_assistant/checkpoint_002000`, which is a BC prior mixed with softmax over search values and therefore has the structure of eq. (2) |
| $V^e_h(s,g)$ | new goal-conditioned value network, input = full observation including goal, output scalar in $[0,1]$ via sigmoid, trained with TD target (13) |
| $X_h(s)$ | new goal-agnostic network, input = observation with goal masked, output positive scalar via softplus, trained with target (14). The sum over $\mathcal G_h$ is replaced by the mean over goals (drop the $|\mathcal G_h|$ factor), which only rescales $X_h$ by a constant. The paper requires $\pi_r$ to be invariant to common rescaling of $V^e$ and eq. (8) with a power-law policy delivers that, so this is behaviour-preserving |
| $U_r(s')$ | replaces `SampleBatch.REWARDS` for the assistant in the training batch, following the pattern of `MbagGAIL._replace_rewards_with_discriminator_scores` |
| $\pi_r$, $V_r$, $Q_r$ | Phase 2 actor-critic. The paper allows "a network approximation of $\pi_r$ trained on $V_r$ with entropy regularisation" in place of the power-law policy. We use the existing `MbagPPO` assistant with entropy regularisation. The MCTS variant is a later branch |

### 3.1 Deviations from the paper, stated explicitly

1. **Human prior from data, not from eqs. (1) to (3).** MBAG human models
   are trained from human demonstrations (BC) or BC plus search (piKL). We do
   not solve eq. (1) with the cautious $\min_{a_r}$; the assistant has no
   commitment actions, so $\mathcal A_r(s)$ never shrinks and the min has no
   lever to act on. The heuristic human is the limiting case
   $\nu=1$ with a deterministic $\pi^0_h$.
2. **Goal set is a distribution, not an enumerable partition.** $X_h$ is a
   mean over sampled goals. With CraftAssist houses the set is huge and is
   never enumerated. For the exact-solver tests in Section 5 the set is
   enumerated.
3. **Mid-episode goal resampling** ($p_g$) is an optional env flag
   `goal_change_prob`, default `0.0`. Off for the first experiments because
   the frozen human models were trained with fixed goals.
4. **Policy form.** Entropy-regularised PPO rather than eq. (5). The paper
   sanctions this for the actor-critic variant.

### 3.2 Zeroing the assistant's goal reward

`MbagEnv._step_player` adds `goal_dependent_reward` for every player. A new
per-player reward key `goal_reward_scale` (default `1.0`) multiplies it. The
assistant config sets it to `0.0`. The MCTS simulator (`MbagEnvModel`) also
applies it so that env and planner agree. `own_reward_prop=1` for the
assistant so it does not receive the human's goal reward through sharing.
The result is that the assistant's env reward is exactly `0` at every step
and the only nonzero reward it ever sees is $U_r$.

## 4. Components

### 4.1 `mbag/power/metrics.py` (base branch)

Pure numpy / torch functions, no RLlib dependency:

- `x_from_attainment(v_e, zeta)`: eq. (7) as a mean over the goal axis.
- `power_bits(x)`: eq. (10).
- `robot_reward(x, xi, eta)`: eq. (8) for one or many humans.
- `soft_power_policy(q, beta)`: eq. (5).
- Constants `PAPER_HYPERPARAMS = dict(zeta=2.0, xi=1.0, eta=1.1, gamma_h=0.99, gamma_r=0.99)`.

### 4.2 `mbag/power/exact.py` (base branch)

Backward induction for eqs. (1) to (9) on small tabular, acyclic or
finite-horizon stochastic games with one robot and $n$ humans. Input: a
`TabularGame` dataclass (states, per-player action sets, transition kernel,
goal sets, terminal flags) plus the human-model parameters
$(\nu, \pi^0_h, \beta_h, \mu_{-h})$ and $(\zeta,\xi,\eta,\beta_r,\gamma_h,\gamma_r)$.
Output: $Q^m_h, \pi_h, V^m_h, Q_r, \pi_r, V^e_h, X_h, W_h, U_r, V_r$ as arrays.

This is the ground truth against which learned estimates are checked and
the place where "exactly the equations" is testable.

### 4.3 `mbag/power/gridworld.py` (base branch)

The paper's key-and-door gridworld from Section 4.2 as a `TabularGame`.
Used as a test: the exact solver must produce a robot policy that fetches
the key, unlocks the door, and steps out of the way.

### 4.4 Environment changes (base branch)

- `RewardsConfigDict.goal_reward_scale` and default `1.0`.
- `MbagConfigDict.goal_change_prob`, default `0.0`. When nonzero, at each
  step with that probability the env draws a new goal from the generator
  and updates the goal observation for all players. Off in all first
  experiments.
- Info dict gains `goal_completed: bool` for the step in which
  `goal_percentage` first reaches `1.0`.

### 4.5 `mbag/rllib/human_power.py` (ppo-assistant branch)

- `AttainmentValueModel` $V^e(s,g)$: reuses the existing convolutional
  backbone with goal channels present, sigmoid head.
- `PowerModel` $X_h(s)$: same backbone with goal channels masked, softplus
  head.
- `MbagHumanPowerPPO(MbagPPO)`: in `training_step`, before the PPO update,
  (i) build $(s, g, s', \text{done}, U_h)$ tuples for the assistant's batch
  using the human's observation from `other_agent_batches` for the goal;
  (ii) TD-update $V^e$ with target (13); (iii) update $X_h$ with target (14)
  using the current $V^e$; (iv) compute $U_r(s')$ for every transition and
  overwrite `SampleBatch.REWARDS` for the `assistant` policy; (v) recompute
  advantages exactly as GAIL does. Metrics logged: mean $W_h$, mean $U_r$,
  $V^e$ loss, $X_h$ loss.
- Config keys: `power_zeta`, `power_xi`, `power_eta`, `power_gamma_h`,
  `power_lr`, `power_num_sgd_iter`, `power_target_update_freq`.

### 4.6 Train config `iccea_power_assistant` (ppo-assistant branch)

Derived from `ppo_assistant`: `run="MbagHumanPowerPPO"`, `mask_goal=True`,
`goal_loss_coeff=0` (no goal inference), `own_reward_prop=1`,
`per_player_goal_reward_scale=[1, 0]`, `gamma=0.99`, paper power
hyperparameters, and a `cpu_smoke` variant with `world_size=(5,5,5)`,
`goal_generator="random"` restricted to a handful of blueprints,
`horizon=100`, `num_workers=2`, heuristic human.

### 4.7 Metrics and evaluation (both branches)

- `MbagCallbacks`: per-episode mean `human_power_bits` and `robot_reward`
  when present in infos.
- `mbag/evaluation/metrics.py`: same two keys, plus `human_actions_to_completion`.
- `evaluate.py` gains a `power_checkpoint` option so offline evaluation can
  report $W_h$ along a trajectory.

## 5. Testing

Base branch:
- Exact solver on a 2-state bandit reproduces the closed forms in
  Section 2.1 footnotes: $k$ deterministic options give $W_h=\log_2 k$; $k$
  uniformly random outcomes with $\zeta=2$ give $W_h=-\log_2 k$.
- Exact solver reproduces Appendix A inequality $E^\zeta \le W$ numerically
  on random bandits.
- Gridworld: robot policy under the exact solver executes key, door, move
  aside.
- Env: `goal_reward_scale=0` for player 1 yields zero goal-dependent reward
  for that player in env and in `MbagEnvModel`, and the existing
  `test_predicted_rewards_equal_rewards_in_alpha_zero` still passes.
- Env: `goal_completed` fires exactly once per completed episode.

PPO branch:
- Reward replacement test: after one `training_step` on a tiny world, the
  assistant batch rewards equal `robot_reward(X_h(s'))` computed directly
  from the `PowerModel`, and no reward equals the env's goal reward.
- Smoke training run of two iterations on CPU completes and logs the four
  power metrics.

## 6. Branch layout

```
master
└── human-power/base           spec, mbag/power/*, env plumbing, metrics, docs/human-power/README.md
    ├── human-power/ppo-assistant   Phase 2 with PPO, train config, CPU experiments
    └── human-power/mcts-assistant  U_r at MCTS nodes via PowerModel (after ppo branch has results)
```

Each branch adds `docs/human-power/<branch>.md` with equations, config
knobs, exact commands, and a results log.

## 7. Out of scope for now

- RunPod or any cluster setup.
- Reproducing the AssistanceZero paper's human study.
- Multi-human MBAG.
- The partially observed formulation (paper Appendix C).
