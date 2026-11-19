# Culture at level 3 — a language-fed `function_metaconcept`

Status: **DESIGN** (2026-11-19). Nothing built. Written for `robocomp/active_inference`, to sit next to
`ring_metaconcept/SCHEMA_GENERALITY_TODO.md`; drafted on `pbustos/active_inference` because that is the
repository this session can write. Companions: `CONCEPT_AGENT_RECIPE.md` §6 (the metaconcept chassis),
`kitchen_metaconcept/etc/config.toml` (the authored class priors this replaces), `language_agent/`
(the VLM), and the paper draft `inner_speech_section.tex` (Eq. 5, the calibrated proposer prior).

Paths below are relative to `robocomp/active_inference` unless stated.

---

## 0. The one idea

Geometry is authored up to level 2; **culture enters at level 3.**

Levels 1–2 already build the scene bottom-up and imagine it top-down without any language: concept agents fit
objects, metaconcepts (`ring`, `kitchen`) fit arrangements over those beliefs, exist by evidence against an explicit
null, and push cavity-corrected priors back down, including next-best views at empty slots. What no level can do is
say **what a room is for** — kitchen, bedroom, office — because that is not a geometric fact. It is a cultural one,
and the VLM is the robot's only source of culture.

So: a level-3 `function_metaconcept`, built on the same chassis rules as levels 1–2 (evidence against a null, no
thresholds, cavity, bounded down-priors), whose **conditional probabilities come from the VLM**, calibrated and
progressively replaced by the robot's own counts. The VLM never runs inside the loop: it **supplies priors and
proposes structure**; C++ evaluates. The same pipeline also gives levels 1–2 what their configs already ask for —
measured instead of hand-elicited class priors — and gives the language agent a model-level trigger for its focused
mode.

---

## 1. What already exists (do not rebuild)

| Mechanism | Where | What level 3 reuses |
|---|---|---|
| Graph-reader front end, main-thread polling, no `update_node`/`update_edge` signals | recipe §6 Delta 1; both metaconcepts | the whole front end |
| Measurements = peers' posteriors, noise = each peer's own published Σ / existence | `ARCHITECTURE.md` §9 | members' existence `L` and class |
| Existence by log Bayes factor vs an explicit null, persistent-evidence birth | recipe §6 "Detection without a threshold" | the existence rule, verbatim |
| Cavity / leave-one-out; EP message down-date | `ring_metaconcept` `compute_cavity_priors`; `kitchen_belief.h` `down_date` | the accounting for any message sent down |
| EMA on `log_odds` ("redundant re-observation must not manufacture confidence") | ring chassis | the same discipline for repeated members |
| Class-membership log-odds, explicitly flagged as AUTHORED and to be replaced by corpus counts | `kitchen_metaconcept/etc/config.toml` lines 56–82 | Phase 0 replaces these |
| `room_explored_p`: P(continuing to explore no longer pays) | `room_concept`, read by `controller/src/mission_explored.h` | coverage term for absence evidence |
| Exploration drive (afford_room + door looks; doors as curiosity) | `controller/src/exploration_drive.h` | the consumer of level-3 exploration targets |
| Per-instance epistemic cycle / NBV | `common/epistemic_step`, each agent's `EpistemicPlanner` | consumer of expected-object targets |
| Person-confirmed claims, posted only after a yes | `language_agent/src/human_claim.h`, `ltom_agent` | a labelled source of room functions for calibration |

---

## 2. Five uses of the VLM, simplest first

1. **Class priors for existing metaconcepts.** `kitchen_metaconcept`'s `MemberClassLogOdds` are elicited by hand
   (`cabinet 0.2, refrigerator 2.0, counter 2.5, hood 3.0, microwave 1.5, chest of drawers −0.5`). The config says
   they "should be replaced by counts from a labelled scene corpus when one exists". The VLM is that corpus, with
   the caveats of §5. *No runtime change.*
2. **Schema proposals.** A new typed arrangement on an existing schema is a config file today (`coffee_cluster`,
   `meeting_set`). The VLM proposes `(schema, anchor class, member class, yaw offset, slot σ)` for arrangements it
   expects ("bed + two nightstands → facing-pair"); the chassis accepts or ignores them by evidence.
3. **Level-3 function** — this note's main design (§3).
4. **Model-level surprise → structural proposals.** Metaconcept residuals become triggers for the language agent's
   focused mode, which proposes a revision of the *model*, not of its parameters (§4).
5. **Open-vocabulary evidence for classes without a producer** (`counter`, `hood`, `microwave` have no node, per the
   kitchen config). VLM `look()` as a low-precision measurement. Riskiest; last.

---

## 3. `function_metaconcept`

### 3.1 Latent

For each room `r` and each function `f ∈ F` (`kitchen, dining, living, bedroom, bathroom, office, corridor`),
an independent hypothesis "`r` serves `f`" against the null "`r` is a generic space". **Not a categorical over F**:
open-plan rooms serve several functions, and one hypothesis per function is exactly how the chassis already treats
metaconcept instances (each has its own evidence vs null). A room with no function above the null is a legitimate
outcome, not a failure.

### 3.2 Evidence

Members are level-1 objects (class `c`, existence probability `p_c = σ(L_c)`) and level-2 arrangements
(`dining_set`, kitchen frame, with their existence). For each class, **presence**, not count, is the observation:
ten chairs must not argue ten times (the same common-mode discipline as the log-odds EMA and the residual `tanh`
saturation).

With `π_cf = P(c present | f)`, `π_c0 = P(c present | null)` and `d_c` the probability that `c` would have been
detected in `r` had it been present:

```
Λ_f(r) = Σ_c  p_c · log( π_cf / π_c0 )
            + (1 − p_c) · log( (1 − π_cf · d_c) / (1 − π_c0 · d_c) )
```

- The first term is evidence from what is there. The second is **expected-but-absent** evidence, and it is
  only as strong as the search was: with `d_c = 0` (nothing looked at) it vanishes, which is the
  negative-information rule already used for instance death (`expected_visible`) and residual carving, lifted to
  the scene level.
- `d_c` starts as `room_explored_p(r) · δ_c` with a per-class detector rate `δ_c`; the detectability model
  (`P(detect | viewpoint, belief)`, Paper 1) replaces it when it lands.
- The soft-presence expectation is an approximation (exact under hard presence); flagged here, not hidden.
- `Λ_f(r)` is the hypothesis's existence log-odds; birth and death follow the chassis's persistent-evidence process.
  No threshold is introduced.

### 3.3 Where `π` comes from — VLM pseudo-counts, then experience

Each `π_cf` gets a Beta prior whose mean is the VLM's elicited probability and whose **strength is earned**:

```
π_cf ~ Beta( κ·β_f · π_cf^VLM ,  κ·β_f · (1 − π_cf^VLM) )
posterior after n_f rooms confirmed as f, of which n_cf contained c:
π̂_cf = ( κ·β_f·π_cf^VLM + n_cf ) / ( κ·β_f + n_f )
```

- `κ` is a small pseudo-count (the VLM is worth a few rooms, not a hundred). `β_f ∈ [0, 1]` is set from the
  VLM's measured calibration (§5.3). This is the conjugate, implementable form of Eq. 5 in the paper draft.
- **Confirmed rooms** come from the person (a `human_claim`-style "this is the kitchen", stored only after their
  yes) or from repeated visits whose function posterior has saturated. Early on the VLM dominates; experience takes
  over as `n_f` grows.
- `π_c0` (the null) is elicited the same way ("probability that a generic room contains a `c`").

### 3.4 Messages down

Phase 1 sends **nothing** into member beliefs. Phase 2 sends exploration targets only. Phase 3, if earned, seeds
level-2 hypotheses.

| Message | Consumer | Contamination risk |
|---|---|---|
| Expected unseen classes `c` with high `π_cf` and low `p_c`, with coarse placement (along walls, near water, near the anchor arrangement) | `exploration_drive.h` / `EpistemicPlanner` as **viewpoint value**, never as existence | selection bias only: it changes where the robot looks, not what a look reports |
| "Instantiate schema S with config K" (e.g. kitchen frame along the north wall) | the relevant metaconcept, as a birth seed | real: a seeded member's existence then contains our message → `down_date` required, as in `kitchen_belief.h` |

**The cavity rule binds the VLM too.** If "kitchen" sends the robot to look for kitchen things and the kitchen
posterior then rises on what those looks found, the hypothesis has partly confirmed itself through selection. Phase 2
keeps it honest because absence counts as evidence in exactly the places that were searched (`d_c` is symmetric:
the same search that can find `c` can fail to). Phase 3 adds the down-date for anything injected into a member.

### 3.5 Node and ownership

- One node per room, `function_<room>`, metaconcept type, carrying `function_labels`, `function_log_bf` (Λ per
  function) and the coverage it was computed at. New attributes need a `REGISTER_TYPE` line and a cortex reinstall:
  **ask Pablo**.
- **Owner: a separate pure-C++ agent**, not `language_agent`. The VLM is not in the runtime loop: the prior table is
  data, the inference is deterministic, and the agent keeps running when the VLM is down.
- Where the table lives: Phase 0–1, an `etc/function_priors.toml` produced by an offline elicitation tool (§5) and
  **reviewed and committed like any authored prior**, with provenance in the file. Agents do not read each other's
  files (structure-from-evidence global constraint), so an online path in which `language_agent` elicits new
  entries goes through the graph (a published prior-table node), Phase 4 at the earliest.

---

## 4. Model-level surprise and the language agent

Levels 1–2 already compute the signals that say **the model is wrong**, as opposed to uncertain. They are the
natural inputs to the language agent's demand signal (paper draft Eq. 6):

| Signal | Source | What the focused voice proposes |
|---|---|---|
| Arrangement `log_bf` collapsing | ring / kitchen CSVs and nodes | wrong schema; the arrangement was moved; ask |
| A member with a large cavity residual (disagrees with its arrangement) | `compute_cavity_priors` / `down_date` | not a member; moved; misclassified |
| A class persistently near zero membership | kitchen membership log | **missing structure**: the live fridge read membership 0.00 until the `Tall` tier existed — a person diagnosed it; this is the voice's job |
| Two schemas both weak and unresolved | `ring_partition` model selection | a schema not in the library (propose one, §2.2) |
| Λ_f expected-but-absent mass high in an explored room | `function_metaconcept` | wrong function; an occluded area worth a view |
| Member yaw contradicting the arrangement (chair facing out) | **not scored yet** — `SCHEMA_GENERALITY_TODO.md` §2.5 | only culture explains it (someone sitting, cleaning) |

**Proposals are text, decisions are evidence.** In Phase 3 the voice's structural proposals (a tier, a schema
config, a class prior) are written for a person to review and apply. Self-modification of agent configs is out of
scope until the review path has a record.

---

## 5. Eliciting priors from the VLM

### 5.1 Protocol
- **Closed vocabulary.** Classes are exactly the graph's class names (the `object_class_of` mapping), functions
  exactly `F`. The VLM is never asked to name something; it is asked a probability about named things.
- **One question per (class, function)**: "In a typical home, what is the probability that a room used as a
  `<f>` contains at least one `<c>`?" plus the null question. Temperature 0, and three paraphrases; keep the
  median, record the spread.
- **Clamp** the resulting log-odds to the range the kitchen config learned the hard way: a strong prior also makes
  a badly fitting member of that class argue hard against the whole hypothesis (its clutter floor falls from
  −1.77 nats at +0.2 to −3.91 at +2.5; the first live kitchen run pinned at the −8 clamp).
- **Provenance in the file**: model tag, date, prompt hash, per-entry spread.

### 5.2 First use (Phase 0)
Run the protocol for the kitchen's seven member classes and compare with the hand-authored `MemberClassLogOdds`
and with the live membership logs. No runtime change; it measures whether the VLM's culture agrees with the
author's and with the data.

### 5.3 Calibration
`β_f` comes from rooms whose function is known: the Webots apartment (ground truth) first, then person-confirmed
rooms. Score the elicited `π` against observed presence (Brier per function); `β_f` falls where the VLM is poorly
calibrated. Expect cultural bias — the VLM's "typical home" is not every home — and let the counts correct it.

---

## 6. Phases and falsifiable checks

| Phase | What | Check (policy vs policy in one closed loop where it applies) |
|---|---|---|
| **P0** | Elicit kitchen class priors (§5.2) | Agreement with authored values; does swapping them in change any live membership decision? |
| **P1** | `function_metaconcept` **read-only**: fits and writes a CSV, creates no node (the kitchen's M2 pattern) | the functions above the null are the right ones in each Webots room; each `Λ_f` moves away from 0 as `room_explored_p` rises; **remove the fridge in sim**: the kitchen hypothesis weakens but survives on the rest |
| **P2** | Node published; exploration targets to the drive | Time and views until every object in a room is found; objects missed in rooms counted as explored (`room_explored_p ≥ 0.9`), with vs without targets |
| **P3** | Model-level surprise → language agent proposals (§4) | **Seeded structural defects**: re-create the two-tier kitchen; does the voice propose a `Tall` tier? Remove a schema; does it propose one the chassis then accepts? |
| **P4** | Online priors through a graph node; schema seeds to level 2 with down-date | Calibration (`β_f`) improves with confirmed rooms; no self-confirmation (seeded vs unseeded posteriors agree on held-out evidence) |

A failure at P1 (function not recoverable from presence and absence) ends the line before any runtime coupling.
A failure at P2 means culture does not pay for exploration and the targets go; the priors may still be worth P0.

---

## 7. Risks

- **Self-confirmation through selection** — §3.4; contained by symmetric `d_c` in P2 and the down-date in P4.
- **Overconfident culture** — Beta strength `κ·β_f`, the clamp, calibration against known rooms.
- **Cultural mismatch** — the VLM's typical home vs this home; the counts correct it, and nothing is ever excluded
  (log-odds down-weight, never forbid).
- **Vocabulary drift** — classes are the graph's names, fixed; a new class needs a new elicitation, not a guess.
- **Open plan** — handled by one hypothesis per function (§3.1), not by a forced choice.
- **Thresholds creeping in** — none is introduced here; flag any that appears, at its site (CLAUDE.md).

## 8. Questions for Pablo

1. Should level 3 be one `function_metaconcept`, or should "function" become one more instance type on a generalised
   metaconcept chassis (the open question at the end of `SCHEMA_GENERALITY_TODO.md`)?
2. Which agent id and DDS domain; the new attributes' `REGISTER_TYPE` lines.
3. Is a person-confirmed room function a `human_claim` kind (stored by `ltom_agent`), or does it belong to `ltsm_agent`
   with the room's identity?
4. Is P0 worth doing before the kitchen frame is allowed to steer cabinets (it is read-only at milestone 2), so that
   what steers them uses measured rather than authored priors?
