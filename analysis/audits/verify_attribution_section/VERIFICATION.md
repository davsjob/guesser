# Attribution section: verification against the final numbers (2026-10-02)

**Scope:** steps 0–6 of the verification plan. Data: MegaScale test, forward only, 8-seed ensemble means, S_Std_A.
**σ_e:** σ_e = σ_intra = √⟨β/(ν(α−1))⟩ (column `intra_epistemic_std`).
**CIs:** 95% protein bootstrap unless noted. 2,000 draws for means; 300–500 draws for correlations and shares.
**Code and raw output:** `main_paper/audits/verify_attribution_section/`. The scripts are `part_a_mu.py`, `part_a2_dssp.py`, `part_b_sigma.py` and `part_c_doubles_limits.py`; each writes a `results_*.json`. They read the attribution_v3 cache and tables and write nothing else.
**Reproduction check:** every original value re-derived here matched the v3 / deep-dive tables exactly. This covers the driving shifts, burial shifts, class means, site share, per-rule cell means, within-class ρ and the 6.7× drop-off. The CIs match to ±0.003.

## Bottom line

- **Step 0 passes.** The σ-IG target is the seed mean of log σ_s from the EDL head alone. It tracks log σ_intra at ρ = 0.995 (singles) and 0.990 (doubles), against 0.67 and 0.31 for σ_inter. The uncertainty paragraph does **not** need to be recomputed wholesale.
- **Correction to the plan.** `epistemic_std` is not σ_inter. It is √(σ_intra² + σ_inter²), the old total σ_e. Its mean is 0.45 against 0.41 for σ_intra in singles, and 0.44 against 0.34 in doubles. Only the deep-dive numbers that read this column change: strand-Pro σ, to-Pro class σ, σ by burial, and the AUC.
- **Result.** Of 27 checks, 22 pass as they stand or need only a wording change. Five need a number updated, and Fig. 4E needs a fix (all listed at the end). None of the qualitative claims reverses.

## Step 0: provenance

| Check | Result | Verdict |
|---|---|---|
| σ-IG target | g = mean_s log σ_s, exact to 1e-15. ρ(g, log σ_intra) = 0.995 / 0.990; ρ(g, σ_inter) = 0.67 / 0.31 | ✅ |
| Forward, 8 seeds, test | 28,172 singles and 18,574 doubles, which are the forward test-set sizes. 8 members per variant; mapping and WT identity check 100% | ✅ |
| Burial | Relative SASA (Tien 2013 max ASA), cutoffs 0.20 / 0.50. n = 7,625 / 11,078 / 9,469 | ✅. The old LaTeX table captions still say "RSA ≤ 20 Å²". Change to "RSA < 0.20" if those tables move to the SI |
| IG completeness | Max \|error\|: μ 0.016 (singles) / 0.078 (doubles); log σ 0.010 / 0.026. μ_null = −0.008 ± 0.057 (singles), 0.002 ± 0.153 (doubles) | ✅ |

## Step 1: driving forces (μ), singles, experimental tiers (stab < 0, destab > 1)

| Claim | Result | Verdict |
|---|---|---|
| Hydrophobicity > clashes > volume > polar solvation | Shifts: +0.37 [0.31, 0.45], +0.25 [0.20, 0.30], +0.22 [0.17, 0.28], +0.12 [0.10, 0.15]. All exclude 0. **Clashes − volume = +0.03 [−0.06, +0.11]**: the full order holds in only 72% of draws. Hydrophobicity first and polar solvation fourth hold in 99% | ⚠️ Change the wording to "hydrophobicity, then clashes and volume (indistinguishable), then polar solvation" |
| "94% on the mutated residue" | Mutation-IG self share is 0.94, which is true by construction. Environment IG per node falls **6.7× [6.4, 7.0]** from the site to > 9 Å | ✅ Phrase it as the environment drop-off. Defined as the mean over variants of each shell's per-node share; pooling mass over variants instead gives 8.4× |
| SASA is the top environment feature | Top by mean \|attr\| in all six dataset × subset sets (MS-S, MS-D, S669, Ssym, FireProt-HF, PTMUL-D) | ✅ |
| "+0.20 core, −0.19 surface" | These are the **whole environment term**, not SASA: core +0.20 [0.12, 0.28], **boundary −0.31 [−0.38, −0.23]**, surface −0.19 [−0.24, −0.14]. The SASA part alone is +0.53 [0.44, 0.61] / −0.15 / −0.09 | ⚠️ Attribute the numbers to "the environment term". Don't imply a monotone gradient: the boundary is the most negative |

## Step 2: burial gating (Fig. 4A)

| Claim | Result | Verdict |
|---|---|---|
| Hydrophobicity core vs surface | Core +0.54 [0.44, 0.64], boundary +0.14 [0.11, 0.18], surface +0.01 [−0.02, 0.03]. The ratio CI is [−473, +548] | ✅ but **drop "77×"**. Say "vanished at the surface", or give the difference, +0.53 [0.43, 0.65] |
| Polar desolvation about 8× lower at the surface | +0.18 [0.15, 0.22] → +0.024 [0.016, 0.035]. Ratio **7.6× [5.2, 11.2]** | ✅ "about 8-fold (95% CI 5–11)" |
| Volume at every depth, U-shaped, more for cavities | Core +0.35 [0.26, 0.45], boundary +0.24 [0.18, 0.31], surface +0.35 [0.24, 0.48]. Cavity +0.71 vs overpacking +0.17. Max \|pred − exp\| per ΔV bin = **0.042** (cavity bin) | ✅ "within 0.05 kcal/mol" (0.042 rounds to 0.04, but is above it) |
| Electrostatics only when the charge changes | Mean \|Electrostatics attr\|: 0.091 [0.076, 0.107] charge-changing vs 0.007 [0.005, 0.008] charge-preserving, **13.7× [11.3, 16.3]**. Charge-changing variants are 45% of variants and carry **92%** of the attribution. Neutral→neutral 0.005. The ratio holds in every burial class: 23× core, 15× boundary, 9× surface | ✅ "almost only (92% of the attribution)". The input change is also 14× larger (1.40 vs 0.10), so the attribution follows the input |

## Step 3: class rules (Fig. 4D)

| Claim | Result | Verdict |
|---|---|---|
| Pro → ring + backbone; Gly → volume; hydrophobic→polar → hydrophobicity | to Pro (n 1,309): Cyclic +0.54 [0.50, 0.58], Backbone entropy +0.20 [0.18, 0.21], H-bonds +0.17 [0.16, 0.19]. to Gly (n 1,314): Volume +0.79 [0.73, 0.84]. hydrophobic→polar (n 3,215): Hydrophobicity +0.56 [0.51, 0.61]. charge introduced (n 3,265): Hydrophobicity +0.48, Polar solvation +0.25. All classes span 28/28 proteins | ✅ Fig. 4D reads the same table; the n labels match |
| Pro penalty by secondary structure (helix +0.75 vs coil +0.28) | **Exposed sites only** (boundary + surface); "helix" means **helix interior**. Helix interior +0.75 [0.69, 0.81] (n 247, 24 proteins) vs coil +0.28 [0.25, 0.31] (n 407). Difference +0.47 [0.42, 0.52]. Helix N-term +0.42 [0.36, 0.47] (n 125); strand +0.53 | ✅ Say "helix interior" and "exposed" |
| DSSP re-implementation | Three-state agreement with the wwPDB HELIX/SHEET records: 1PSE 0.99, 1TUC 0.95, 2K5H 0.94. Strand recall 0.93–0.96, precision 0.92–1.00. Ideal α-helix: all H except the terminal residues. Isolated extended chain: all coil | ✅ Two caveats. No α-rich natural protein could be tested, because RCSB is not reachable from here. Multi-model NMR files are read with the last model's coordinates; this is negligible (98% agreement with model 1 for 2K5H) |

## Step 4: uncertainty attribution (Fig. 4B, C, E), with σ_e = σ_intra

| Claim | Result | Verdict |
|---|---|---|
| Site baseline = 66% of var(log σ), singles | **66% [61, 70]**. Doubles 47% [23, 72] (protein) / [26, 66] (66 site pairs) | ✅ |
| Site part couples to \|μ\|, mutation part doesn't | ρ(g0, \|μ\|) = **+0.60 [0.57, 0.63]**; ρ(g − g0, \|μ\|) = **−0.06 [−0.12, +0.01]**. σ_e itself: 0.58, the same value as in the selective-prediction section | ✅ |
| Doubt added at the boundary (+0.006 / +0.05 / +0.04 per rule) | The cell means reproduce: core +0.006 [−0.001, 0.012], boundary +0.048 [0.041, 0.054], surface +0.037 [0.032, 0.042]; boundary − core [0.035, 0.049]. The order survives using all 23 features (0.004 / 0.038 / 0.030), variant weighting (−0.009 / 0.032 / 0.020) and leaving out any one feature (boundary 0.030–0.056). **However,** the magnitudes depend on which cells are included, and the boundary is the maximum for only 13 of 25 individual rules | ⚠️ Replace with the selection-free version. Mean mutation part g − g0: **core −0.09 [−0.16, −0.03], boundary +0.30 [0.24, 0.36], surface +0.19 [0.14, 0.23]**; boundary − surface +0.12 [0.08, 0.15]. In σ, that is ×0.91 / ×1.35 / ×1.21. Say "on average" |
| A clear structural change lowers σ (within-class ρ(A_μ, A_σ)) | Clashes **−0.27 [−0.32, −0.21]**, electrostatics **−0.14 [−0.23, −0.07]**, H-bonds **−0.18 [−0.24, −0.12]**. Also rotamer violations −0.20; hydrophobicity is opposite, +0.29 | ✅ |
| Strand Pro: σ highest (1.06), MAE highest (1.12) | 1.06 was the old total. Exposed strand to-Pro: **σ_e 1.02 [0.92, 1.13]**, MAE 1.12 [0.91, 1.37]. It is still the highest of the four SS states for both (next: coil σ_e 0.81, MAE 0.82) | 🔧 **1.06 → 1.02** |
| (related) to-Pro is the most uncertain class (σ_e 0.84) | **0.80 [0.74, 0.86]**; still the most uncertain class (next: hydrophobic→hydrophobic 0.51) | 🔧 **0.84 → 0.80** |
| (related) σ by burial | σ_intra: core 0.58, boundary 0.40, surface 0.28 (old total: 0.64 / 0.44 / 0.32) | 🔧 if quoted |

**Fig. 4E needs a fix.** `fig4_rules_trust.py` panel E sums `class_mean_attr_log_sigma_epi.csv`, which has only 12 of the 23 features. The axis label says "mutation part", which is exactly g − g0 (all 23 features, by completeness). The bars understate it by up to 0.08. Some examples (12-feature sum → g − g0): to Pro 0.84 → 0.91, to Gly 0.20 → 0.26, charge reversal 0.16 → 0.24, charge introduced 0.14 → 0.18. Charge reversal moves up past from-Gly. Plot the class mean of `gm` instead. The values are in `results_part_b.json` → `class_sigma`.

## Step 5: doubles

| Claim | Result | Verdict |
|---|---|---|
| Volume share of μ variance 37% vs 19% | Doubles **37% [19, 49]** (protein) / [24, 50] (site pair); singles **19% [16, 22]**. In doubles, Volume becomes #1, ahead of hydrophobicity (22%) | ✅ Give the CIs. The protein-bootstrap CIs just touch |
| Compression to about half the sum of singles; head ≈ constant offset (4%) | Base + phys attribution mass is 0.47–0.51 of the summed singles in every distance bin. Head variance share **4% [−3, 12]**. Mean head offset **+0.78**, which matches the doubles section | ✅ |
| Mutation part lowers σ in doubles (−0.36) | **−0.36 [−0.45, −0.28]** (66 site pairs) / [−0.49, −0.23] (20 proteins). Negative in 79% of pairs | ✅ It can stay a main result, not just "supporting" |

## Step 6: limits

| Claim | Result | Verdict |
|---|---|---|
| Seed stability 0.99 / 0.98 aggregate, 0.68 / 0.26 per variant | **The two aggregate numbers used different metrics.** μ: rank agreement of mean \|attr\|. σ: rank agreement of class-balanced variance shares. On one metric, mean \|attr\|: **μ 0.99, σ 0.99** (singles); 0.99 / 0.98 (doubles). Class-balanced shares: μ 0.88, σ 0.98. Per variant (same metric for both): 0.68 / 0.26 ✓ | 🔧 Use one metric. Suggested wording: "Dataset-level feature rankings agreed across seeds (ρ = 0.99 for both μ and σ), per-variant rankings did not (0.68 for μ, 0.26 for σ)" |
| σ attributions don't detect errors better than σ (AUC 0.57 → 0.59) | Recomputed with σ_intra: high-σ_e quartile within protein, wrong (\|err\| > 1) vs right (< 0.5), protein-grouped 5-fold CV. Singles: σ_e + its two parts + μ + burial **0.58 → + σ attributions 0.60** (+ μ attributions 0.61). Doubles: 0.46 → 0.55 (+ class × burial) → 0.57 | 🔧 **0.57 → 0.59 becomes 0.58 → 0.60**. Conclusion unchanged |

## Text and figure changes, in section order

1. Driving forces: hydrophobicity first; clashes and volume tied; polar solvation fourth.
2. "94%" → the environment-IG drop-off, 6.7× (95% CI 6.4–7.0), with the definition stated.
3. "+0.20 / −0.19" belongs to the environment term, not SASA. The boundary is −0.31.
4. Drop "77×". Say "vanished at the surface (+0.01, CI −0.02 to 0.03)".
5. Polar desolvation: "about 8-fold (5–11)".
6. Volume: "within 0.05 kcal/mol in every ΔV bin".
7. Electrostatics: "almost only when the charge changes: 92% of the attribution, 14× per variant".
8. Pro × SS: "helix interior vs coil, exposed sites".
9. Boundary doubt: use the mean mutation part by burial (−0.09 / +0.30 / +0.19), not the per-rule cell means.
10. Strand Pro σ_e 1.06 → **1.02**. To-Pro class σ_e 0.84 → **0.80**.
11. Seed stability: one metric for both (0.99 / 0.99; per variant 0.68 / 0.26).
12. AUC 0.57 → 0.59 becomes **0.58 → 0.60**.
13. **Fig. 4E:** plot the class mean of g − g0 (all 23 features), not the 12-feature sum.
