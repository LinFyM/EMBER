# EMBER Prior–Innovation Writer Design

Status: next canonical fresh architecture, derived after the Core–Program
macro0→200 non-pass and its non-rollout internal analysis.

This document defines one executable architecture. It is not a collection of
optional patches. The v6 upstream representation is retained because the
available evidence says that it already separates semantic invariants from
ordered video evidence; the complete Core/Procedure compiler is replaced
because that is where the measured signal collapse occurs.

## 1. Decision

The Writer shall treat task LoRA generation as:

```text
stable task-semantic prior
+ video-conditioned ordered innovation
→ coordinated public-LoRA slots
```

It shall not treat Core and Procedure as symmetric factors whose elementwise
product licenses every LoRA contribution.

The new compiler therefore has exactly four stages:

1. route the semantic Core into the 320 public-LoRA identities;
2. use each routed Core prior to read only the time-varying part of Procedure;
3. add that ordered innovation to the semantic prior;
4. coordinate the already formed slots and decode the sealed rank-16 LoRA.

There is no bilinear product, AdaLN modulation, learned scalar gate, manual
branch scale, residual escape path, contrastive loss, or additional trainable
adapter.

## 2. Evidence that determines the redesign

The fresh Core–Program Writer trained with the same task-complete B20,
fast-decay400 contract as the strongest v6 run. Its fixed correct400 curve was:

```text
macro50 / 100 / 150 / 200
84      / 75      / 60      / 76
```

This is far below v5.2 step900 `132/400` and v6 macro200 `133/400`. Its
four-checkpoint per-task envelope was only `95/400`, so the failure is not
explained by choosing the wrong single checkpoint.

The macro50 internal analysis found:

- shuffled/reversed relative differences were already large in Procedure:
  `.571/.775`;
- they shrank to `.0288/.0446` in effective LoRA and `.00669/.00995` in
  policy action;
- centered Procedure AC carried much stronger order contrast than its DC
  component, but reading the complete raw Procedure let DC dominate content;
- the compiler sent about `2.7×` more gradient per coordinate to Core than
  Procedure;
- a constant Procedure produced almost the same LoRA norm as the real
  Procedure;
- Core–Program and the preceding Procedure-only Recenter architecture both
  remained near `85/400`.

The root cause is therefore not that the video encoders failed to observe
order. It is that two successive architectural axioms removed the semantic
output basis or buried the dynamic innovation:

```text
Recenter:
  Procedure must create all LoRA content
  → semantic-basis starvation

Core–Program:
  Core and raw Procedure must multiply
  → raw-DC dominance, moving-basis coupling, weak Procedure transfer
```

v5.2 simultaneously achieved `132/400` and
`correct/same/wrong/shuffled/reversed = 132/138/74/82/83`. It is direct
evidence that a useful Core path is compatible with strong video causality.
Consequently, direct Core contribution is not itself a forbidden shortcut.
The required boundary is that video contributes a grounded, ordered change to
that prior and that invalid order cannot masquerade as the same innovation.

## 3. Retained upstream representation

No upstream branch is redesigned in this experiment.

### 3.1 Shared frozen-policy evidence

For every sampled video frame and the task language, the Writer retains:

- the text-only task queries `Q_text`;
- multimodal task-token evidence `M_f`;
- task-queried patch evidence `G_f`;
- the native 50-position Action-Expert suffix, reduced only after the frozen
  Action Expert forward to the per-frame action probe `A_f`.

The Writer still receives no teacher action, state, reward, task ID, filename,
or hidden normalization statistic.

### 3.2 Semantic Core

The v6 Semantic Core remains:

```text
mean multimodal backbone
+ task-selected centered frame residual
```

It is permutation-invariant to the frame set and remains aligned to the task
token axis. It represents task language, semantic roles, objects, relations,
and stable scene context.

### 3.3 Ordered Procedure

For each arm, visual transitions are recomputed in that arm's actual frame
order:

```text
D_0 = 0
D_f = G_f - G_(f-1)
```

The action probe queries these task-grounded transitions, the result is added
to the action probe, and a two-block causal RoPE Procedure encoder produces:

```text
P ∈ R^(F×256)
```

This retains the two necessary Procedure streams: what the source policy
believes is being done and what task-relevant semantic effect follows.

## 4. Prior–Innovation compiler

Let `S=320` be the sealed public-LoRA slot count and `d=256`.

### 4.1 Routing identities

The existing learned query, module, layer, and rank identities form:

```text
R ∈ R^(S×d)
```

Routing identities may affect addressing through queries and keys. They never
enter an attention value path and therefore cannot invent video-conditioned
content.

### 4.2 Semantic prior

The routed Core reader has learned, bias-free `Q/K/V/O` projections:

\[
\widehat C_s =
W_o^C\,\operatorname{Attention}\left(
W_q^C R_s,\,
W_k^C\operatorname{RMSNorm}(C),\,
W_v^C C
\right)
\]

\[
B_s=\operatorname{RMSNorm}(\widehat C_s)
\]

`B_s` is the stable semantic prior for one exact public-LoRA row. A valid Core
may produce a useful LoRA without Procedure, because deeper task understanding
can legitimately improve the frozen source policy. It is not expected to
contain the video-taught execution innovation.

### 4.3 Procedure innovation

First compute a masked, per-video time mean in FP32 and center only the value
content:

\[
\bar P =
\frac{1}{|V|}\sum_{f\in V}P_f
\]

\[
P_f^\Delta =
\begin{cases}
P_f-\bar P,&f\in V\\
0,&f\notin V
\end{cases}
\]

The keys retain the complete normalized Procedure and true sampled-frame
positions. The values contain only the centered innovation:

\[
U_s =
W_o^P\,\operatorname{Attention}\left(
\operatorname{RoPE}_0(W_q^P B_s),\,
\operatorname{RoPE}_{pos(f)}
  (W_k^P\operatorname{RMSNorm}(P_f)),\,
W_v^P P_f^\Delta
\right)
\]

All `Q/K/V/O` projections are learned, bias-free, width 256 with eight heads.
The query comes only from `B_s`: routing already selected the Core content, and
must not provide a second task-independent query shortcut.

This is not uniform temporal averaging. The complete Procedure determines
where each slot reads, while the value path prevents Procedure DC from
overwhelming ordered change.

### 4.4 Additive fusion

\[
Z_s=B_s+U_s
\]

Addition expresses the actual causal role: the teaching video changes a
pre-existing task-semantic hypothesis. Core and Procedure are not forced into
a symmetric product, so each has a direct, stationary gradient path.

No learned scalar controls their ratio. Their vector-valued projections learn
which semantic coordinates are prior and which are innovation.

### 4.5 Slot coordination and LoRA decoding

One residual slot block coordinates the 320 already formed slots:

```text
address = RMSNorm(Z) + R
Q/K     = learned projections of address
V       = learned projection of Z only
Z       = Z + attention(Q,K,V)
Z       = Z + FFN(RMSNorm(Z))
H       = RMSNorm(Z)
```

Routing remains address-only. The final RMSNorm provides one stable interface
to the unchanged width-256 factor heads. The factor heads still start with
zero final projections, so every fresh Writer generates the exact identity
public LoRA at step 0.

## 5. Structural invariants

The implementation and focused tests must establish:

1. `Core=0` implies `B=0`. The Procedure query is then zero, its attention is
   uniform, and the mean of the centered linear values is zero; therefore
   `U=0` and the compiler has no Procedure-only content path.
2. A temporally constant Procedure has `P^Δ=0`, so it adds no innovation while
   retaining the valid semantic prior.
3. The same Core with shuffled or reversed frames retains the same `B` but may
   change `U`, final slots, effective LoRA, and policy action.
4. Routing identities affect Q/K addressing only, never V content.
5. Invalid padded frames contribute neither to the mean nor to attention.
6. Centering is computed in FP32 before casting back, including BF16
   non-power-of-two sequence lengths.
7. Fresh public LoRA is exactly the sealed identity template.
8. After factor heads open, gradients reach Core, centered Procedure,
   transition, and both compiler paths with finite values.

These are structural statements, not desired rollout outcomes.

## 6. Capacity

With width 256 and eight heads, the compiler budget is:

| component | parameters |
|---|---:|
| routing identities and routing norm | 91,648 |
| Core `Q/K/V/O` reader, memory norm, and prior norm | 262,656 |
| Procedure `Q/K/V/O` innovation reader and memory norm | 262,400 |
| residual slot block and output norm | 787,200 |
| **compiler total** | **1,403,904** |

Replacing the `1,665,792`-parameter Core–Program compiler yields the
implementation-enumerated Writer total:

```text
10,905,856 - 1,665,792 + 1,403,904
= 10,643,968
```

The exact module enumeration and trainable contract both confirm this total.
The reduction is a consequence of removing an unnecessary width-512 bilinear
product, not an attempt to minimize capacity. Every necessary attention keeps
full-width learned Q/K/V/O projections.

## 7. Experiment contract

The architecture uses a fresh incompatible schema and cannot resume any
earlier Writer checkpoint.

To isolate the architectural hypothesis, its first segment keeps the strongest
v6 training contract unchanged:

```text
GPU4–7, four DDP ranks
24 tasks exactly once per macro
rank-local long-first, rotating physical-rank assignment
one teacher video and B20 independent action queries per task
task mean, then 24-task equal weighting
one sync / clip / AdamW / scheduler step per macro
peak LR 3e-4, warmup 17, cosine decay through macro400, floor 1e-5
fresh macro0→200, checkpoint every25
```

Before formal training it needs its own longest-video B20 stability profile
and exact-resume smoke. The fixed correct400 checkpoints are
macro50/100/150/200 with the existing identical no-replacement schedule.

Decision gates:

- below the contemporaneous one-hour v5.2/v6 range, especially `60–90`:
  no second hour and no behavior-specificity rollouts; run internal transfer
  analysis and redesign from the measured bottleneck;
- recover roughly `125–133` with broad task coverage:
  the compiler hypothesis survives and training conflict becomes the next
  object of study;
- equal or exceed the prior single-checkpoint region and still rise:
  exact-resume a second hour;
- only after stable performance is near `145` or correct400 reaches `150`:
  run correct/wrong/shuffled/reversed behavior rollouts.

Checkpoint fusion, averaging, and outcome-based task or video selection remain
forbidden.

## 8. Falsifiable predictions

If the redesign fixes the measured bottleneck:

- first-hour absolute performance returns at least to the v5.2/v6 range;
- success covers at least six of eight validation tasks rather than merely
  migrating among a few;
- centered Procedure differences survive into effective LoRA and policy
  action substantially better than in Core–Program;
- constant-Procedure and Core-prior LoRAs remain meaningful but no longer
  explain essentially the entire real-video output;
- Procedure/Core gradient transfer is no longer structurally suppressed by
  the bilinear product.

If Procedure remains large while LoRA/action differences again collapse, the
compiler hypothesis is falsified. If transfer is healthy but correct400
remains low, the next diagnosis moves upstream or to the AS objective rather
than adding a fusion scale.
