# Experiment C: observations

Observations only, numbered in the order they were made; interpretation is in
`LOG.md` and the paper. Superseded observations are marked in place and keep
their numbers.



1. The vocabulary stretches to fill the room it is given: trained tables
   spread over 91-98% of d at every size, needed or not.
2. The input side has slack once the room is big enough: small rooms are
   fully used; past a point part of the room can be removed without loss,
   and the share used falls as d grows at fixed vocabulary (1.0 down to
   about 0.6-0.7).
3. The output side never has slack: it uses all of d in every model, size
   and vocabulary tested.
4. Used input room grows with exposure: through training it keeps climbing
   and has not stopped at 300B tokens.
5. Used input room grows with model size at fixed vocabulary, slower than
   d (about 1.7K, 2.8K, 3.1K at d = 2.5K, 4K, 5K).
6. A bigger vocabulary uses more room at the same d: core about 1.5K for
   32K and 50K vocabularies; all 2048 for 100K.
7. Token vectors are, on average, about as separate as random vectors in
   the same room (input: mean pairwise overlap 1.1-1.5x random), but almost
   every token has a close neighbour (nearest-neighbour overlap 3-6x
   random). The crowding is local, not general. (`interference.py`,
   `results/interference.md`; d = 2048 models.)
8. The output side is more crowded than the input side, most of all among
   tokens that occur in ordinary text (mean overlap 2.6-2.7x random for
   TinyLlama and Pythia, 7.7x for OLMo-2).
9. Among tokens that occur in ordinary text, input nearest-neighbour
   crowding rises with vocabulary size: 3.2x (32K), 4.1-4.2x (50K), 4.8x
   (100K).
10. Through training (pythia-2.8b input table), spread does not grow
    with the directions needed. From the random start, training first
    concentrates the table (share of d covered: 0.99 at step 0 to 0.89 at
    step 33,000; directions for 90% of variance: 0.85 d to 0.65 d) while
    nearest neighbours pull close (0.09 to 0.49). After that, spread edges
    back up (0.89 to 0.91) and crowding levels off (about 0.51), while the
    directions needed keep rising (observation 4). (`through_training.py`,
    `results/parked/through_training.md`.)
11. The tail (input directions past the loose-tolerance core) carries
    token-specific content, not only spread. Giving each token another
    token's tail, or matched random values, costs at least as much as
    zeroing the tail in every model, and much more in the largest (12b:
    zeroed +0.10, shuffled +0.29, random +0.30 bits per byte). Wrong tail
    values are worse than none. Restoring row lengths after zeroing helps
    in three models and hurts in one (2.8b). (`semantic_core.py`,
    `results/semantic_core.md`.)
12. For each token's nearest neighbour in the core, the tail pushes the
    pair apart most when they are surface variants of one word (case,
    leading space), less for shared stems, least for different words, in
    every model (12b: 0.110, 0.094, 0.075 in cosine). The push grows with
    model size. The pairs it separates most are punctuation and function
    words that the core nearly merges ('.'/',', ' was'/' were',
    ' the'/' to', ' 0'/' 1').
13. Nearest neighbours get steadily further apart as directions are added
    (median cosine about 0.70 at 64 directions to 0.21-0.35 at full d),
    with no break at the core boundary. At the same number of directions,
    the larger models are emptier.
14. Token identity needs few directions. Identity information
    (definitions.md section 1; `identity_info.py`,
    `results/identity_info.md`) at epsilon 4, dictionary view: the input
    tables carry 6-14 of 15.6 bits in their top 64 directions and nearly
    all of it by 256 to 1024, far below the thousands of directions the
    loss-based test needs (observations 2, 5).
15. Surface-variant information fills earlier than between-word
    information. By the half-fill dimensions (definitions.md 1.3), the
    within-class part reaches half its full-room value first in 78 of 96
    cases (8 models x 2 tables x 2 views x 3 resolutions), at the same k in
    the other 18, and never later (epsilon 4, dictionary: within at 8-64
    directions, between at 32-128; output tables at 8-16).
16. Efficiency (identity carried over the spectrum's capacity ceiling) falls
    as d grows: at epsilon 4, dictionary view, input table 0.67 at d = 512
    down to 0.07 at d = 5120; the output tables match within 0.01-0.04.
17. At the same number of directions, the input table carries more identity
    than the output table (epsilon 4, dictionary, 64 directions: 9.5 vs 7.2
    bits at 1b, 14.3 vs 9.2 at 12b).
18. At the same number of directions, larger models carry more identity
    (input, epsilon 4, dictionary, 64 directions: 6.2 bits at 70m to 14.3
    at 12b). Part of this comes from the noise scale being set per model
    (definitions.md 1.2), so cross-size comparisons at fixed epsilon are
    not like for like.
19. No real crossover. With surface-variant classes, the crossover
    (definitions.md 1.3) appears in 23 of 144 cases (Pythia, TinyLlama,
    OLMo-2; 2 tables, 2 views, 3 resolutions), and in all 23 it falls after
    both parts have saturated, where both per-direction gains are within
    two standard errors of zero. Wherever either part is still growing,
    each direction adds more between-word bits than within-word bits.
    (`identity_info_xvocab.py`, `results/identity_info_xvocab.md`.)
20. The half-fill order holds across vocabularies. In TinyLlama (32K) and
    OLMo-2 (100K) as in Pythia (50K), the within-class part fills first or
    at the same k, never later. The output tables put surface-variant
    information in their top 8 directions in all three families (epsilon 4,
    dictionary); the input tables reach half of it at 32 (TinyLlama,
    Pythia) to 64 (OLMo-2) directions.
21. With grammatical-role classes (12-tag universal part of speech, tagged
    in context; `identity_roles.py`, `results/identity_roles.md`), in the
    text view grammatical role fills before the word within the role: role
    first in 53 of 60 cases (10 models x 2 tables x 3 resolutions), tied in
    7, never later. At epsilon 4, role reaches half its value by 8-16
    directions (OLMo-2 input: 32); the word within the role by 32-64.
22. In the dictionary view (every token with a known role counted equally;
    roles from a lexicon, covering 34-48% of each vocabulary) there is no
    consistent order: role first in 11 of 60 cases, tied in 27, word first
    in 22.
23. With role classes the crossover appears in all 120 cases, at 8-16 (or
    16-32) directions, never within sampling noise. Role carries about 2-3
    bits and the word within a role about 7-12, so absolute within-class
    gains overtake at once. With surface-variant classes (observation 19)
    the reverse holds and no real crossover appears.
24. Pinned noise (definitions.md 1.10, derived before computing;
    `pinned_noise.py`, `results/pinned_noise.md`). Residual-stream
    interference, the context-driven part of what the first block adds to
    a token's vector, gives eps_res = 2.2-2.7 for 70m-410m and 3.5-4.2 from
    1b up. Counting the token-determined part too gives 8.1-20.8. The fp16
    storage floor gives 2.1e-4.
25. At eps_res, identity information reaches 99% of its value at k = d by
    128-512 directions in all eight Pythia input tables (dictionary view;
    `identity_pinned.py`, `results/identity_pinned.md`). At eps = 8 the same
    point is 1,280-1,792 directions (512-1,024 for the three smallest).
26. On its own, the input tail (directions past the k*(0.1) core) resolves
    most of the token's identity: at eps_res 10.4 of 15.6 bits (1.4b), 14.0
    (2.8b), essentially all (6.9b, 12b). Nearly all of it is shared with
    the core: the redundancy R is within 0.01 bits of I(tail) at eps_res;
    at eps = 8 the tail adds 0.46 (1.4b), 0.35 (2.8b), 0.06 (6.9b) and 0.03
    (12b) bits beyond the core. (`tail_identity.py`,
    `results/tail_identity.md`.)
27. The tail's class structure matches a shuffled tail. Wherever the tail
    does not already resolve the token (1.4b and 2.8b at eps 4, 8, res;
    6.9b and 12b at eps 8), the true tail's between-class information is
    within two standard errors of all five shuffled tails for
    surface-variant classes in every case, and for role classes in all but
    one: in 12b at eps 8 the true tail carries less role information than
    shuffled (-0.5 to -2.6 standard errors). Gaps above two standard errors
    appear only for role classes at eps = 2 in 1.4b and 2.8b, where the
    tail already resolves 99.6-99.9% of identity, and amount to about 0.01
    bits. Where the tail resolves all of it (6.9b and 12b at eps 2, 4 and
    res; every model at the fp16 floor) the comparison has no power.
    Method: the "two standard errors" combine the true and shuffled errors
    as if independent (hypot of the two), although both share the token
    and noise draws; the error of their difference should come from paired
    per-draw differences. The decision rule was one-sided (above only).
    The true tail's I_B is below all five shuffled tails in three
    surface-view cases: 1.4b eps 4 (9.396 vs 9.400-9.545), 1.4b eps_res
    (9.559 vs 9.560-9.706), 12b eps 8 (13.047 vs 13.081-13.160); and in
    three role-view cases: 1.4b eps 4 (0.8596 vs 0.8604-0.9106), 1.4b
    eps_res (0.893 vs 0.895-0.946), 12b eps 8 (1.754 vs 1.768-1.823).
    *Superseded.* This test measured the tail beyond k*(0.1). A follow-up
    on the block between k_0.99 and k*(0.01) was planned and dropped
    (2026-10-09): the open question moved to what the grammar needs (see
    the ideas entry "Identity is the vocabulary half").
28. A Gaussian tail with the true tail's per-direction variances resolves
    more identity than the true tail in the surface view (1.4b, eps 4:
    10.62 vs 10.26 bits; 12b, eps 8: 14.09 vs 13.95), in every case where
    the tail does not already resolve the token. In the role view the
    random tail resolves less than the true tail in 1.4b (eps 4, 8, res),
    2.8b eps 8 (4.839 vs 5.110), 6.9b eps 8 (10.389 vs 10.624) and 12b
    eps 8 (13.349 vs 13.416), and more in 2.8b eps 4 (13.678 vs 13.634) and
    eps_res (13.487 vs 13.463). The random tail uses all-token variances,
    while p in the role view covers only role tokens. One random draw per
    case. The true tail is not isotropic, and its structure lowers
    distinguishability; by observation 27 that structure is not
    surface-variant or role structure.
29. Nearest neighbours in the core have tails more alike than chance.
    Predicting held-out tokens' tail coordinates from their nearest
    neighbours in the core gives R^2 well above the shuffled-tail baseline
    in every model (k = 5: +0.036, +0.032, -0.056, -0.054 against -0.200 to
    -0.201 for 1.4b, 2.8b, 6.9b, 12b; k = 20: +0.013, +0.011, -0.017, -0.013
    against -0.050; spread across permutations under 0.005). In absolute
    terms the core predicts almost none of the tail: every R^2 is within
    0.06 of zero. (`nonlinear_echo.py`, `results/superseded/nonlinear_echo.md`.)
    *Superseded.* This test measured the tail beyond k*(0.1). A follow-up
    on the block between k_0.99 and k*(0.01) was planned and dropped
    (2026-10-09): the open question moved to what the grammar needs (see
    the ideas entry "Identity is the vocabulary half").
30. A small MLP from core to tail does no better than on a shuffled tail
    (held-out R^2 -0.072 vs -0.068, -0.044 vs -0.044, -0.053 vs -0.044,
    -0.044 vs -0.035). Ordinary least squares is below its shuffled
    baseline (-0.067 vs -0.036 at 1.4b), as expected: principal coordinates
    are uncorrelated over the whole vocabulary, so any correlation fitted on
    the training tokens is reversed on the held-out ones. The same
    train/held-out reversal lowers any fitted predictor's held-out R^2 on
    the true tail relative to a shuffled tail, so "no better than
    shuffled" bounds a nonlinear echo only up to that bias. The MLP's
    true-tail R^2 is below all five shuffled tails at 1.4b (-0.0716 vs
    -0.0677 to -0.0672), 6.9b (-0.0530 vs -0.0439 to -0.0436) and 12b
    (-0.0438 vs -0.0347 to -0.0344).
    *Superseded.* This test measured the tail beyond k*(0.1). A follow-up
    on the block between k_0.99 and k*(0.01) was planned and dropped
    (2026-10-09): the open question moved to what the grammar needs (see
    the ideas entry "Identity is the vocabulary half").
31. Over the evaluation-text tokens (3,536), pairwise cosines in the core
    and in the tail are weakly anti-correlated (Spearman -0.024 to -0.032:
    -0.0235, -0.0301, -0.0315, -0.0325), against shuffled-tail baselines
    within 0.001 of zero. The shuffled tail is not a matched baseline: zero
    core-tail cross-covariance over the vocabulary pushes the true pairing
    toward negative values even without structure. A matched null (a
    Gaussian table with the same spectrum, or a shuffle re-orthogonalized
    to the core) has not been run.
    *Superseded.* This test measured the tail beyond k*(0.1). A follow-up
    on the block between k_0.99 and k*(0.01) was planned and dropped
    (2026-10-09): the open question moved to what the grammar needs (see
    the ideas entry "Identity is the vocabulary half").
32. Identity in the text view at the pinned noise (`identity_pinned_text.py`,
    `results/identity_pinned_text.md`). With the noise held at the same
    absolute sigma_res, the text view's own scale is 8-10% below the full
    table's, so eps in text terms is 2.39-4.67 (full-table eps 2.20-4.20).
    The text view has about 3,500 token types and H(V) = 9.43 bits; I(d) at
    sigma_res is 99.83-100% of it in every model and basis.
33. In the unweighted (squash-test) basis at sigma_res, k95 / k99 / k99.9
    are: 70m 384 / 512 / 512; 160m 384 / 768 / 768; 410m 128 / 384 / 768;
    1b 768 / 1,536 / 1,984; 1.4b 512 / 1,024 / 1,536; 2.8b 384 / 768 or
    1,024 / 1,536; 6.9b 256 / 512 / 1,280; 12b 256 / 384 / 768. Against
    k*(0.01): k99 is below it in 1b (1,536 vs 1,984), 1.4b (1,024 vs 1,664),
    2.8b (768-1,024 vs 1,664), 6.9b (512 vs 2,816), 12b (384 vs 3,072) and
    410m (384 vs 1,024, with k* taken as d); it equals k* (= d) in 70m (512)
    and 160m (768). In 1b, k99.9 equals k*. I(k*) is at least 99.9% of I(d)
    in every model; the largest gain past k* is 0.0093 +/- 0.0028 bits (1b).
34. In the frequency-weighted basis at sigma_res, k99 is 128-256 in all
    eight models (k95 64-256, k99.9 256-384).
35. At the fp16 floor, k99 is 8 directions in every model and both bases.
    Draws: 4,000 by default; 16,000 where the 99% point could not be
    separated from the neighbouring grid point (410m basis A; 160m and 12b
    basis B); 64,000 for 2.8b and 6.9b basis A. At 64,000, 2.8b's k99 is
    still unresolved between 768 and 1,024 (margin +0.0023 +/- 0.0022 bits
    at 768).
36. Gaussian capacity of the core and tail (`tail_capacity.py`,
    `results/tail_capacity.md`), C = sum 0.5 log2(1 + lambda_i / sigma^2)
    over each block's unweighted principal directions, sigma = eps * s.
    C(core) is 20-492 bits across models and eps (78-168 at eps_res),
    above H(V) = 15.62 everywhere; I(core) is 3-73% of C(core). C(tail) at
    eps_res is 11.3 (1.4b), 18.5 (2.8b), 57.5 (6.9b) and 83.5 (12b) bits.
    Tail power is 12%, 17%, 25% and 31% of the table's total.
37. Where C(tail) is below H(V) (1.4b at eps 4, 8 and eps_res; 2.8b and
    6.9b at eps 8), the measured I(tail) in the surface view is 92-99% of
    C(tail): 1.4b 0.92 / 0.97 / 0.92, 2.8b 0.99, 6.9b 0.93. Where C(tail)
    is above H(V) but I(tail) is still short of H(V) (2.8b at eps 4 and
    eps_res, 12b at eps 8), it is 73-77% of C(tail).
38. In the role view at eps 8, I(tail) exceeds the all-token C(tail) in
    1.4b (2.878 +/- 0.046 vs 2.81) and 2.8b (5.110 +/- 0.061 vs 4.89); it
    is within 0.4 standard errors of the capacity from the role tokens'
    own covariance (2.86 and 5.09). No measured I exceeds that capacity by
    more than 0.4 standard errors anywhere.
39. Shannon's count, turned around (`capacity_count.py`,
    `results/capacity_count.md`): n_flat = 2 H(T) / log2(1 + 1/eps^2), with
    eps = sigma_res / s and s the view's own scale, is 115-393 in the
    dictionary view (H(T) = 15.62) and 81-291 in the text view (H(T) =
    9.43). It does not involve d.
40. (Superseded by 42-43, which measure k99 on a fine grid.) Against the
    measured k99 interval (grid point below, excluded, to k99) at sigma_res: in the text view's frequency-weighted basis n_flat
    is inside the interval at 410m, 1.4b, 6.9b and 12b, 4% and 14% above it
    at 2.8b and 1b, 5% and 9% below it at 70m and 160m. In the dictionary
    view it is inside at 160m, 410m, 1b and 1.4b, below at 70m (174 vs
    (256, 384]), above at 2.8b, 6.9b and 12b (306 vs (64, 128] at 12b). In
    the squash-test basis on the text it is below the interval in all
    eight, by 1.2-4.4 times at the interval's lower end.
41. The capacity lower bound k_C (smallest k with C(k) >= 0.99 I(d), C
    from each table's own covariance in the view's basis, log-det in the
    squash basis) is 6-21 directions in the weighted basis, 30-113 in the
    squash basis and 36-129 in the dictionary view: 3-43 times below the
    measured k99 (12-43 times in the weighted basis).
42. Fine grid (`identity_fine.py`, rule committed before running in
    9a93675; `results/identity_fine.md`). Text view, frequency-weighted
    basis, sigma_res, k in steps of 16: k95 / k99 / k99.9 are 70m 112 /
    160 / 272; 160m 96 / 144 / 224; 410m 64 / 96 / 144; 1b 160 / 240 / 352;
    1.4b 128 / 208 / 336; 2.8b 112 / 176 / 288; 6.9b 96 / 144 / 224; 12b
    96 / 144 / 208. k99 spans 96-240 while d spans 512-5,120. Dictionary
    view: k99 is 256, 208, 112, 384, 304, 240, 144, 128; k99.9 reaches d in
    1b, 1.4b and 2.8b. The 99% cut did not separate from its neighbour at
    64,000 draws in 70m and 6.9b (text) and 160m (dictionary).
43. Shannon's count fails the rule. Primary test (text view, weighted
    basis, 99%): n_flat / k99 = 0.76, 0.81, 0.84, 1.21, 1.19, 1.52, 1.40,
    1.53 (70m to 12b); 4 of 8 within 0.8-1.25; mean |ln ratio| L = 0.274
    against L0 = 0.199 for the best constant (149), so the count does not
    beat a constant. It also fails at the 95% and 99.9% cuts (L 0.528 vs
    0.189; 0.344 vs 0.227) and in the dictionary view at every cut (99%:
    0.352 vs 0.352; 3 of 8 agree). n_flat ranges 81-291 across the models,
    k99 96-240.
44. Against the fine k99, the capacity lower bound k_C (observation 41) is
    7.6-24 times smaller in the text view's weighted basis and 2.7-3.6
    times smaller in the dictionary view.
45. After the fact, not part of the rule in identity_fine.py: a version of
    Shannon's count rescaled by one fitted factor (c = 0.83) also loses to
    the best constant in the primary test (mean |ln ratio| 0.230 vs 0.199).
    Every constant from about 131 to 194 beats the unscaled count; the best
    constants are 144-160. Computed from results/identity_fine.json.
46. Loss at identity's 99% point, measured directly (`claim_checks.py`,
    rules committed in 6cc55f3 before running; `results/claim_checks.md`).
    Squashing the input table to k99 (text view, squash-test basis,
    sigma_res) costs 0.7767 bits per byte at 410m (k = 384), 0.1387 at 1b
    (1,536), 0.4701 at 1.4b (1,024), 1.3333 at 2.8b (768; 0.8683 at
    1,024), 1.7332 at 6.9b (512) and 1.6094 at 12b (384): above 0.1 in
    every model from 410m up. At 70m and 160m k99 = d (no change).
47. Tail interventions repeated at 1.4b and 2.8b (same script): ten
    shuffles (seeds 1-10) and ten matched random tails (seeds 101-110)
    each. Every draw costs more than zeroing the tail. 1.4b: zeroed
    0.9400; shuffled 0.9514-0.9766 (mean 0.9620); random 0.9451-0.9746
    (mean 0.9551). 2.8b: zeroed 0.8776; shuffled 0.8915-0.9145 (mean
    0.8984); random 0.8840-0.9017 (mean 0.8921).

