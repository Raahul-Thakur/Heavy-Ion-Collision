# Scientific Validation Plan

This project is an educational ALICE Open Data analysis pipeline. It is useful
for reproducible exploration, but the raw outputs must pass additional checks
before they are described as scientifically validated physics results.

## Dataset Provenance

The current AO2D example target is CERN Open Data record 11537:

- Dataset: LHC15o_000245349_Pb-Pb_5.02TeV_2409842
- Collision system: Pb-Pb
- Collision energy: 5.02 TeV
- Run: 245349
- Recorded: 2015
- Published: 2025
- DOI: 10.7483/OPENDATA.ALICE.Q4RZ.YFRX
- Record URL: https://opendata.cern.ch/record/11537
- Open-access analysis software: https://github.com/AliceO2Group/O2OpenAccess

The CERN Open Data record says the sample contains 2,409,842 events across
1,438 files. A local single-file or few-file run should therefore be treated as
a subset, not the full dataset.

## Current Status

The code currently computes raw reconstructed-track observables:

- event multiplicity after simple pT and eta cuts
- dN/deta density within the selected acceptance
- pT spectra and an invariant-yield proxy
- centrality-binned QA summaries using available O2 centrality tables
- a raw v2{2} two-particle proxy
- same-event Delta eta / Delta phi pair histograms

These outputs are not corrected for tracking efficiency, acceptance, secondary
contamination, fake tracks, event-selection effects, or systematic
uncertainties.

The AO2D loader applies conservative local selections where the needed columns
exist:

- collision vertex window `|z| < 10 cm`
- selected-collision flag bit 16
- Run-2 event-cut mask `1023`
- TPC quality cuts using found clusters, crossed rows, and chi2/cluster

These selections remove obvious non-analysis collision candidates and improve
QA monotonicity, but they do not replace the official ALICE correction chain.

## Required Validation Before Physics Claims

1. Confirm AO2D branch mappings against the exact ALICE O2 table definitions
   for this production:
   - pT from `abs(1 / fSigned1Pt)`
   - eta from `asinh(fTgl)`
   - phi from `fAlpha + asin(fSnp)`
   - centrality from the selected `O2centrun2...` table

2. Apply official-quality event and track selection:
   - collision/event quality
   - primary vertex requirements
   - pileup rejection
   - track quality and primary-track criteria
   - detector acceptance and kinematic cuts matching the reference analysis

3. Add detector corrections:
   - tracking efficiency
   - acceptance
   - secondary-particle contamination
   - bin migration or unfolding if required

4. Compare against ALICE reference measurements for the same system, energy,
   centrality bins, and kinematic cuts:
   - dN/deta
   - pT spectra
   - centrality and multiplicity QA
   - v2 or flow observables

5. Quantify uncertainties:
   - statistical uncertainties
   - systematic variations of cuts and correction assumptions
   - event-sample dependence

## Validation Report

Every analysis run writes `validation_report.csv`. Treat any `warning` or
`fail` row as a blocker for claiming publication-quality scientific validity.

The report is intentionally conservative: it can show that the software ran and
that basic QA checks passed, but only external reference comparisons and
corrections can establish final scientific validity.

## Bundled Reference Comparison

The repository includes
`references/alice_pbpb_5020_dndeta_table1_full.csv`, transcribed from the
ALICE paper "Centrality dependence of the charged-particle multiplicity density
at mid-rapidity in Pb-Pb collisions at sqrt(sNN) = 5.02 TeV" (Phys. Rev. Lett.
116, 222302; arXiv:1512.06104). The paper reports primary charged-particle
`dNch/deta` values in `|eta| < 0.5`.

The file `references/alice_pbpb_5020_dndeta_midrapidity.csv` contains the
subset and width-weighted combinations matching the project's current
centrality bins: 0-5%, 5-10%, 10-30%, 30-50%, and 50-80%.

The project compares each available centrality bin by averaging its raw
`dN/deta` output over `|eta| < 0.5` and writing `reference_comparison.csv`.
This is deliberately strict about provenance but conservative about
interpretation: a passing row means the raw output is numerically near the
published reference, not that detector corrections and systematic
uncertainties have been completed.

If the comparison fails while internal QA checks pass, do not tune scale factors
to force agreement. The next valid step is to implement or import the official
efficiency, acceptance, secondary-contamination, and fake-track corrections for
the same data production and selection.
