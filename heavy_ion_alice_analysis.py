"""
Heavy-Ion Collision Analysis (ALICE Open Data)

This script reads ALICE ROOT files with uproot/awkward and produces a small
heavy-ion analysis pipeline:
- QA plots for eta, pT, and event multiplicity
- normalized dN/deta and pT spectra
- centrality-binned dN/deta, pT spectra, and average multiplicity
- a starter elliptic-flow proxy v2{2} from two-particle azimuthal correlations
- two-particle Delta eta / Delta phi angular-correlation histograms
- CSV outputs for downstream plotting or reporting

Example:
  python heavy_ion_alice_analysis.py --data_dir /path/to/alice/rootfiles --max_files 5
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm


# -------------------------
# Config
# -------------------------

CENTRALITY_BINS: List[Tuple[str, float, float]] = [
    ("0-5%", 0.0, 5.0),
    ("5-10%", 5.0, 10.0),
    ("10-30%", 10.0, 30.0),
    ("30-50%", 30.0, 50.0),
    ("50-80%", 50.0, 80.0),
]

DATASET_PROVENANCE = {
    "record": "CERN Open Data record 11537",
    "doi": "10.7483/OPENDATA.ALICE.Q4RZ.YFRX",
    "system": "Pb-Pb",
    "energy": "5.02 TeV",
    "run": "245349",
    "period": "LHC15o",
    "recorded": "2015",
    "published": "2025",
}

REFERENCE_DNDETA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "references",
    "alice_pbpb_5020_dndeta_midrapidity.csv",
)


@dataclass
class TrackCuts:
    pt_min: float = 0.15
    pt_max: float = 50.0
    eta_abs_max: float = 0.8
    vertex_z_abs_max: float = 10.0
    require_collision_flag16: bool = True
    event_cuts_mask: int = 1023
    require_tpc_quality: bool = True
    tpc_min_found: int = 70
    tpc_min_crossed_rows: int = 70
    tpc_max_chi2: float = 4.0


@dataclass
class AnalysisConfig:
    eta_bins: int = 60
    pt_bins: int = 80
    pair_eta_bins: int = 80
    pair_phi_bins: int = 72
    max_pair_events: int = 500
    max_pair_tracks: int = 250
    random_seed: int = 7


@dataclass
class BranchMap:
    pt: str
    eta: str
    phi: Optional[str] = None
    cent: Optional[str] = None
    event_id: Optional[str] = None


# -------------------------
# ROOT discovery utilities
# -------------------------

def list_trees_and_branches(root_path: str, max_items: int = 20) -> Dict[str, List[str]]:
    """Inspect a ROOT file and return {tree_name: [branch_names...] }."""
    out: Dict[str, List[str]] = {}
    with uproot.open(root_path) as f:
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, "keys") and hasattr(obj, "arrays"):
                branches = list(obj.keys())
                out[str(key)] = branches[:max_items]
    return out


def guess_tree_candidates(root_path: str) -> List[str]:
    """Return candidate TTrees likely to contain event/track branches."""
    candidates: List[Tuple[str, int]] = []
    with uproot.open(root_path) as f:
        for key in f.keys():
            obj = f[key]
            if not (hasattr(obj, "keys") and hasattr(obj, "arrays")):
                continue
            branches = set(map(str, obj.keys()))
            score = 0
            for token in ["pt", "pT", "eta", "phi", "fPt", "fEta", "fPhi"]:
                score += sum(1 for name in branches if token.lower() in name.lower())
            if score > 0:
                candidates.append((str(key), score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in candidates[:5]]


def ao2d_dataframe_prefixes(root_path: str) -> List[str]:
    """Return AO2D dataframe prefixes such as DF_... if the file is O2/AO2D."""
    prefixes = set()
    with uproot.open(root_path) as f:
        for key in f.keys():
            name = str(key).split(";")[0]
            if "/" not in name:
                continue
            prefix, tree_name = name.split("/", 1)
            if prefix.startswith("DF_") and tree_name.startswith("O2track"):
                prefixes.add(prefix)
    return sorted(prefixes)


def is_ao2d_file(root_path: str) -> bool:
    return len(ao2d_dataframe_prefixes(root_path)) > 0


def is_aliesd_file(root_path: str) -> bool:
    """Detect classic AliRoot ESD files such as AliESDs.root."""
    if os.path.basename(root_path).lower() == "aliesds.root":
        return True
    try:
        with uproot.open(root_path) as f:
            keys = {str(key).split(";")[0] for key in f.keys()}
            return "esdTree" in keys
    except Exception:
        return False


def pick_branch(branches: List[str], preferred: List[str]) -> Optional[str]:
    """Pick the first available branch, preferring exact then contains matches."""
    lower = {branch.lower(): branch for branch in branches}
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]

    for name in preferred:
        for branch in branches:
            if name.lower() in branch.lower():
                return branch
    return None


def build_branch_map(tree: uproot.behaviors.TTree.TTree) -> BranchMap:
    branches = list(map(str, tree.keys()))

    pt = pick_branch(branches, ["pt", "pT", "fPt", "TrackPt", "fP[3]", "fTPCPt"])
    eta = pick_branch(branches, ["eta", "fEta", "TrackEta"])
    phi = pick_branch(branches, ["phi", "fPhi", "TrackPhi"])
    cent = pick_branch(
        branches,
        ["centrality", "cent", "fCent", "V0M", "V0Mpercentile", "fV0M"],
    )
    event_id = pick_branch(branches, ["event", "evt", "EventID", "fEvent", "run", "fRun"])

    if pt is None or eta is None:
        raise RuntimeError(
            "Could not find pT/eta branches automatically. "
            "Run with --inspect to see available trees/branches and update mappings."
        )
    return BranchMap(pt=pt, eta=eta, phi=phi, cent=cent, event_id=event_id)


# -------------------------
# Data loading
# -------------------------

def load_tracks_from_file(
    root_path: str,
    tree_name: Optional[str],
    cuts: TrackCuts,
    max_events: Optional[int] = None,
) -> Tuple[ak.Array, Optional[ak.Array]]:
    """
    Load tracks from one ROOT file.

    Returns:
      tracks: awkward array with event entries and fields pt, eta, optionally phi
      centrality: event-level or event-like centrality branch if found
    """
    with uproot.open(root_path) as f:
        if tree_name is None:
            candidates = guess_tree_candidates(root_path)
            if not candidates:
                raise RuntimeError(f"No candidate TTrees found in {root_path}")
            tree_name = candidates[0]

        tree = f[tree_name]
        bmap = build_branch_map(tree)

        columns = [bmap.pt, bmap.eta]
        if bmap.phi:
            columns.append(bmap.phi)
        if bmap.cent:
            columns.append(bmap.cent)

        arrays = tree.arrays(columns, entry_stop=max_events)
        pt = arrays[bmap.pt]
        eta = arrays[bmap.eta]
        phi = arrays[bmap.phi] if bmap.phi else None
        cent = arrays[bmap.cent] if bmap.cent else None

        mask = (pt >= cuts.pt_min) & (pt <= cuts.pt_max) & (ak.abs(eta) <= cuts.eta_abs_max)

        out = {"pt": pt[mask], "eta": eta[mask]}
        if phi is not None:
            out["phi"] = phi[mask]

        return ak.zip(out, depth_limit=1), cent


def first_existing_tree(file_obj: uproot.ReadOnlyDirectory, candidates: List[str]) -> Optional[str]:
    keys = set(file_obj.keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def read_ao2d_centrality(
    file_obj: uproot.ReadOnlyDirectory,
    prefix: str,
    n_collisions: int,
    max_events: Optional[int],
) -> Optional[np.ndarray]:
    centrality_candidates = [
        ("O2centrun2v0m", "fCentRun2V0M"),
        ("O2centrun2v0a", "fCentRun2V0A"),
        ("O2centrun2cl0", "fCentRun2CL0"),
        ("O2centrun2cl1", "fCentRun2CL1"),
        ("O2centrun2remu5", "fCentRun2RefMult5"),
        ("O2centrun2remu8", "fCentRun2RefMult8"),
        ("O2fmd", "fCentrality"),
    ]
    stop = max_events if max_events is not None else n_collisions

    for tree_base, branch in centrality_candidates:
        tree_key = first_existing_tree(
            file_obj,
            [f"{prefix}/{tree_base};1", f"{prefix}/{tree_base}_001;1", f"{prefix}/{tree_base}_002;1"],
        )
        if tree_key is None:
            continue
        tree = file_obj[tree_key]
        if branch not in tree.keys():
            continue
        values = tree.arrays([branch], library="np", entry_stop=stop)[branch].astype(float)
        if len(values) >= min(stop, n_collisions):
            return values[: min(stop, n_collisions)]
    return None


def load_ao2d_from_file(
    root_path: str,
    cuts: TrackCuts,
    max_events: Optional[int] = None,
) -> Tuple[ak.Array, Optional[ak.Array]]:
    """
    Load ALICE AO2D/O2 track tables.

    AO2D stores tracks as flat rows linked to collision rows via fIndexCollisions.
    For the barrel O2track table, pT and eta can be derived from fSigned1Pt and
    fTgl. Phi is approximated from the local track parameters fAlpha and fSnp.
    """
    pt_events: List[List[float]] = []
    eta_events: List[List[float]] = []
    phi_events: List[List[float]] = []
    cent_events: List[float] = []
    have_centrality = False

    with uproot.open(root_path) as f:
        prefixes = ao2d_dataframe_prefixes(root_path)
        for prefix in prefixes:
            collision_key = first_existing_tree(
                f,
                [f"{prefix}/O2collision_001;1", f"{prefix}/O2collision;1"],
            )
            track_key = first_existing_tree(
                f,
                [f"{prefix}/O2track;1", f"{prefix}/O2track_001;1"],
            )
            if collision_key is None or track_key is None:
                continue

            collision_tree = f[collision_key]
            n_collisions = int(collision_tree.num_entries)
            if max_events is not None:
                remaining = max_events - len(pt_events)
                if remaining <= 0:
                    break

            collision_columns = ["fPosZ", "fFlags"]
            if "fIndexBCs" in collision_tree.keys():
                collision_columns.append("fIndexBCs")
            collision_arrays = collision_tree.arrays(collision_columns, library="np")
            event_mask = np.ones(n_collisions, dtype=bool)
            event_mask &= np.isfinite(collision_arrays["fPosZ"])
            event_mask &= np.abs(collision_arrays["fPosZ"]) <= cuts.vertex_z_abs_max
            if cuts.require_collision_flag16 and "fFlags" in collision_arrays:
                event_mask &= (collision_arrays["fFlags"].astype(int) & 16) != 0

            bc_key = first_existing_tree(
                f,
                [f"{prefix}/O2run2bcinfo_001;1", f"{prefix}/O2run2bcinfo;1"],
            )
            if (
                cuts.event_cuts_mask
                and bc_key is not None
                and "fIndexBCs" in collision_arrays
                and "fEventCuts" in f[bc_key].keys()
            ):
                bc_cuts = f[bc_key].arrays(["fEventCuts"], library="np")["fEventCuts"].astype(int)
                bc_idx = collision_arrays["fIndexBCs"].astype(int)
                bc_ok = (bc_idx >= 0) & (bc_idx < len(bc_cuts))
                event_mask &= bc_ok
                event_mask[bc_ok] &= (bc_cuts[bc_idx[bc_ok]] & cuts.event_cuts_mask) == cuts.event_cuts_mask

            selected_collisions = np.flatnonzero(event_mask)
            if max_events is not None:
                selected_collisions = selected_collisions[:remaining]
            if len(selected_collisions) == 0:
                continue

            track_tree = f[track_key]
            required = ["fIndexCollisions", "fTgl", "fSigned1Pt"]
            optional_phi = ["fAlpha", "fSnp"]
            missing = [branch for branch in required if branch not in track_tree.keys()]
            if missing:
                raise RuntimeError(f"AO2D track tree {track_key} is missing branches: {missing}")

            arrays = track_tree.arrays(required + optional_phi, library="np")
            coll_idx = arrays["fIndexCollisions"].astype(int)
            signed_inv_pt = arrays["fSigned1Pt"].astype(float)
            tgl = arrays["fTgl"].astype(float)

            valid = np.isin(coll_idx, selected_collisions) & np.isfinite(signed_inv_pt)
            valid &= np.abs(signed_inv_pt) > 1e-12

            pt = 1.0 / np.abs(signed_inv_pt)
            eta = np.arcsinh(tgl)
            valid &= (pt >= cuts.pt_min) & (pt <= cuts.pt_max) & (np.abs(eta) <= cuts.eta_abs_max)

            track_extra_key = first_existing_tree(
                f,
                [f"{prefix}/O2trackextra_002;1", f"{prefix}/O2trackextra_001;1", f"{prefix}/O2trackextra;1"],
            )
            if cuts.require_tpc_quality and track_extra_key is not None:
                extra_tree = f[track_extra_key]
                extra_required = [
                    "fTPCNClsFindable",
                    "fTPCNClsFindableMinusFound",
                    "fTPCNClsFindableMinusCrossedRows",
                    "fTPCChi2NCl",
                ]
                if all(branch in extra_tree.keys() for branch in extra_required):
                    extra = extra_tree.arrays(extra_required, library="np")
                    findable = extra["fTPCNClsFindable"].astype(float)
                    found = findable - extra["fTPCNClsFindableMinusFound"].astype(float)
                    crossed = findable - extra["fTPCNClsFindableMinusCrossedRows"].astype(float)
                    tpc_chi2 = extra["fTPCChi2NCl"].astype(float)
                    quality = (
                        (found >= cuts.tpc_min_found)
                        & (crossed >= cuts.tpc_min_crossed_rows)
                        & np.isfinite(tpc_chi2)
                        & (tpc_chi2 > 0)
                        & (tpc_chi2 <= cuts.tpc_max_chi2)
                    )
                    valid &= quality

            if "fAlpha" in arrays and "fSnp" in arrays:
                snp = np.clip(arrays["fSnp"].astype(float), -1.0, 1.0)
                phi = arrays["fAlpha"].astype(float) + np.arcsin(snp)
                phi = (phi + math.pi) % (2.0 * math.pi) - math.pi
            else:
                phi = np.full_like(pt, np.nan, dtype=float)

            selected_position = {int(collision): pos for pos, collision in enumerate(selected_collisions)}
            event_pt = [[] for _ in range(len(selected_collisions))]
            event_eta = [[] for _ in range(len(selected_collisions))]
            event_phi = [[] for _ in range(len(selected_collisions))]

            for event_index, pt_value, eta_value, phi_value in zip(
                coll_idx[valid], pt[valid], eta[valid], phi[valid]
            ):
                output_index = selected_position[int(event_index)]
                event_pt[output_index].append(float(pt_value))
                event_eta[output_index].append(float(eta_value))
                event_phi[output_index].append(float(phi_value))

            pt_events.extend(event_pt)
            eta_events.extend(event_eta)
            phi_events.extend(event_phi)

            cent = read_ao2d_centrality(f, prefix, n_collisions, n_collisions)
            if cent is not None:
                have_centrality = True
                cent_events.extend([float(cent[index]) for index in selected_collisions])
            else:
                cent_events.extend([np.nan] * len(selected_collisions))

    if not pt_events:
        raise RuntimeError(f"No AO2D collision/track tables could be loaded from {root_path}")

    tracks = ak.zip(
        {
            "pt": ak.Array(pt_events),
            "eta": ak.Array(eta_events),
            "phi": ak.Array(phi_events),
        },
        depth_limit=1,
    )
    cent_array = ak.Array(cent_events) if have_centrality else None
    return tracks, cent_array


def load_aliesd_from_file(
    root_path: str,
    cuts: TrackCuts,
    max_events: Optional[int] = None,
) -> Tuple[ak.Array, Optional[ak.Array]]:
    """
    Load classic AliRoot ESD files, usually named AliESDs.root.

    This reader requires a Python environment with ROOT plus ALICE AliRoot/O2
    libraries available. Plain PyPI uproot cannot generally unpack AliESDEvent
    object graphs into track tables by itself.
    """
    try:
        import ROOT  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "This looks like a classic AliRoot ESD file. Reading AliESDs.root "
            "requires PyROOT with ALICE AliRoot libraries available. Use the "
            "ALICE Open Data VM / AliRoot environment, then run this script "
            "there, or use an AO2D.root file for the pure-Python uproot path."
        ) from exc

    for library in ["libTree", "libSTEERBase", "libESD"]:
        try:
            ROOT.gSystem.Load(library)
        except Exception:
            pass

    if not hasattr(ROOT, "AliESDEvent"):
        raise RuntimeError(
            "PyROOT is available, but AliRoot classes are not loaded. "
            "AliESDs.root needs ALICE AliRoot libraries, especially AliESDEvent."
        )

    root_file = ROOT.TFile.Open(root_path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ESD file with ROOT: {root_path}")

    tree = root_file.Get("esdTree")
    if not tree:
        root_file.Close()
        raise RuntimeError(f"No esdTree found in AliESD file: {root_path}")

    esd = ROOT.AliESDEvent()
    esd.ReadFromTree(tree)
    n_entries = int(tree.GetEntries())
    if max_events is not None:
        n_entries = min(n_entries, max_events)

    pt_events: List[List[float]] = []
    eta_events: List[List[float]] = []
    phi_events: List[List[float]] = []
    cent_events: List[float] = []
    have_centrality = False

    for entry in range(n_entries):
        tree.GetEntry(entry)
        event_pt: List[float] = []
        event_eta: List[float] = []
        event_phi: List[float] = []

        n_tracks = int(esd.GetNumberOfTracks())
        for i_track in range(n_tracks):
            track = esd.GetTrack(i_track)
            if not track:
                continue

            pt = float(track.Pt())
            eta = float(track.Eta())
            phi = float(track.Phi())
            if not np.isfinite(pt) or not np.isfinite(eta) or not np.isfinite(phi):
                continue
            if pt < cuts.pt_min or pt > cuts.pt_max or abs(eta) > cuts.eta_abs_max:
                continue

            event_pt.append(pt)
            event_eta.append(eta)
            event_phi.append(phi)

        pt_events.append(event_pt)
        eta_events.append(event_eta)
        phi_events.append(event_phi)

        centrality_value = np.nan
        try:
            centrality = esd.GetCentrality()
            if centrality:
                centrality_value = float(centrality.GetCentralityPercentile("V0M"))
                have_centrality = have_centrality or np.isfinite(centrality_value)
        except Exception:
            pass
        cent_events.append(centrality_value)

    root_file.Close()

    tracks = ak.zip(
        {
            "pt": ak.Array(pt_events),
            "eta": ak.Array(eta_events),
            "phi": ak.Array(phi_events),
        },
        depth_limit=1,
    )
    cent_array = ak.Array(cent_events) if have_centrality else None
    return tracks, cent_array


def load_dataset(
    data_dir: str,
    pattern: str = "*.root",
    max_files: int = 5,
    tree_name: Optional[str] = None,
    cuts: TrackCuts = TrackCuts(),
    max_events_per_file: Optional[int] = None,
    file_format: str = "auto",
) -> Tuple[ak.Array, Optional[ak.Array]]:
    """Load multiple ROOT files and concatenate event arrays."""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {data_dir} with pattern {pattern}")

    tracks_all = []
    cent_all = []
    for path in tqdm(files[:max_files], desc="Loading ROOT files"):
        selected_format = file_format
        if selected_format == "auto":
            if tree_name is None and is_ao2d_file(path):
                selected_format = "ao2d"
            elif tree_name is None and is_aliesd_file(path):
                selected_format = "aliesd"
            else:
                selected_format = "generic"

        if selected_format == "ao2d":
            tracks, cent = load_ao2d_from_file(path, cuts, max_events=max_events_per_file)
        elif selected_format == "aliesd":
            tracks, cent = load_aliesd_from_file(path, cuts, max_events=max_events_per_file)
        elif selected_format == "generic":
            tracks, cent = load_tracks_from_file(path, tree_name, cuts, max_events=max_events_per_file)
        else:
            raise ValueError("file_format must be one of: auto, ao2d, aliesd, generic")
        tracks_all.append(tracks)
        if cent is not None:
            cent_all.append(cent)

    tracks_cat = ak.concatenate(tracks_all, axis=0)
    cent_cat = ak.concatenate(cent_all, axis=0) if cent_all else None
    return tracks_cat, cent_cat


# -------------------------
# Analysis helpers
# -------------------------

def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def has_field(tracks: ak.Array, field: str) -> bool:
    return field in list(ak.fields(tracks))


def flatten_field(tracks: ak.Array, field: str) -> np.ndarray:
    return ak.to_numpy(ak.flatten(tracks[field], axis=None))


def event_multiplicity(tracks: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.num(tracks["pt"], axis=1))


def centrality_to_numpy(cent: Optional[ak.Array], n_events: int) -> Optional[np.ndarray]:
    """Convert common centrality branch shapes into one value per event."""
    if cent is None:
        return None

    arr = ak.Array(cent)
    if arr.ndim > 1:
        arr = ak.firsts(arr, axis=1)

    values = ak.to_numpy(ak.fill_none(arr, np.nan))
    if len(values) != n_events:
        flat = ak.to_numpy(ak.flatten(cent, axis=None))
        if len(flat) == n_events:
            values = flat
        else:
            return None
    return values.astype(float)


def centrality_mask(cent_values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.isfinite(cent_values) & (cent_values >= lo) & (cent_values < hi)


def histogram_density(values: np.ndarray, edges: np.ndarray, n_events: int) -> pd.DataFrame:
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    per_event_density = counts / max(n_events, 1) / widths
    return pd.DataFrame(
        {
            "bin_low": edges[:-1],
            "bin_high": edges[1:],
            "bin_center": centers,
            "bin_width": widths,
            "counts": counts,
            "per_event_density": per_event_density,
        }
    )


def compute_dn_deta(
    tracks: ak.Array,
    eta_edges: np.ndarray,
    label: str = "all",
) -> pd.DataFrame:
    eta = flatten_field(tracks, "eta")
    df = histogram_density(eta, eta_edges, len(tracks))
    df.insert(0, "centrality", label)
    df.rename(columns={"per_event_density": "dN_deta"}, inplace=True)
    return df


def compute_pt_spectrum(
    tracks: ak.Array,
    pt_edges: np.ndarray,
    eta_abs_max: float,
    label: str = "all",
) -> pd.DataFrame:
    pt = flatten_field(tracks, "pt")
    df = histogram_density(pt, pt_edges, len(tracks))
    pt_center = df["bin_center"].to_numpy()
    eta_width = 2.0 * eta_abs_max
    invariant_yield = df["counts"].to_numpy() / np.maximum(
        len(tracks) * 2.0 * math.pi * pt_center * df["bin_width"].to_numpy() * eta_width,
        1e-12,
    )
    df.insert(0, "centrality", label)
    df.rename(columns={"per_event_density": "dN_dpT_per_event"}, inplace=True)
    df["invariant_yield_proxy"] = invariant_yield
    return df


def v2_two_particle(tracks: ak.Array, event_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute a starter v2{2} proxy using the Q-vector identity:
      c2 = (|sum exp(2i phi)|^2 - M) / (M * (M - 1))
      v2{2} = sqrt(c2), if c2 is positive.
    """
    if not has_field(tracks, "phi"):
        return {"c2": np.nan, "v2_2": np.nan, "events_used": 0, "pairs_used": 0}

    selected = tracks[event_mask] if event_mask is not None else tracks
    c2_weighted_sum = 0.0
    pair_count_sum = 0
    events_used = 0

    for phi_values in ak.to_list(selected["phi"]):
        phi = np.asarray(phi_values, dtype=float)
        phi = phi[np.isfinite(phi)]
        multiplicity = len(phi)
        if multiplicity < 2:
            continue
        q2 = np.exp(2j * phi).sum()
        pairs = multiplicity * (multiplicity - 1)
        c2 = (abs(q2) ** 2 - multiplicity) / pairs
        c2_weighted_sum += c2 * pairs
        pair_count_sum += pairs
        events_used += 1

    if pair_count_sum == 0:
        return {"c2": np.nan, "v2_2": np.nan, "events_used": events_used, "pairs_used": 0}

    mean_c2 = c2_weighted_sum / pair_count_sum
    return {
        "c2": mean_c2,
        "v2_2": math.sqrt(mean_c2) if mean_c2 > 0 else np.nan,
        "events_used": events_used,
        "pairs_used": pair_count_sum,
    }


def sampled_pair_deltas(
    tracks: ak.Array,
    max_events: int,
    max_tracks: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample same-event track pairs and return Delta eta and wrapped Delta phi."""
    if not has_field(tracks, "phi"):
        return np.array([]), np.array([])

    rng = np.random.default_rng(seed)
    n_events = len(tracks)
    event_indices = np.arange(n_events)
    if n_events > max_events:
        event_indices = rng.choice(event_indices, size=max_events, replace=False)

    delta_eta_parts = []
    delta_phi_parts = []

    for idx in event_indices:
        eta = np.asarray(ak.to_list(tracks["eta"][idx]), dtype=float)
        phi = np.asarray(ak.to_list(tracks["phi"][idx]), dtype=float)
        valid = np.isfinite(eta) & np.isfinite(phi)
        eta = eta[valid]
        phi = phi[valid]

        if len(phi) < 2:
            continue
        if len(phi) > max_tracks:
            keep = rng.choice(np.arange(len(phi)), size=max_tracks, replace=False)
            eta = eta[keep]
            phi = phi[keep]

        i_upper, j_upper = np.triu_indices(len(phi), k=1)
        deta = eta[i_upper] - eta[j_upper]
        dphi = phi[i_upper] - phi[j_upper]
        dphi = (dphi + math.pi) % (2.0 * math.pi) - math.pi
        delta_eta_parts.append(deta)
        delta_phi_parts.append(dphi)

    if not delta_eta_parts:
        return np.array([]), np.array([])
    return np.concatenate(delta_eta_parts), np.concatenate(delta_phi_parts)


# -------------------------
# Plots and CSV outputs
# -------------------------

def plot_dn_deta_table(df: pd.DataFrame, out_dir: str, filename: str) -> None:
    plt.figure()
    for label, group in df.groupby("centrality"):
        plt.step(group["bin_center"], group["dN_deta"], where="mid", label=label)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$(1/N_{\mathrm{events}})\ dN/d\eta$")
    plt.title("Charged-particle pseudorapidity density")
    plt.legend()
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


def plot_pt_spectrum_table(df: pd.DataFrame, out_dir: str, filename: str) -> None:
    plt.figure()
    for label, group in df.groupby("centrality"):
        plt.step(group["bin_center"], group["dN_dpT_per_event"], where="mid", label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$p_T$ [GeV/c]")
    plt.ylabel(r"$(1/N_{\mathrm{events}})\ dN/dp_T$")
    plt.title("Normalized transverse-momentum spectra")
    plt.legend()
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


def plot_multiplicity(tracks: ak.Array, out_dir: str, bins: int = 60) -> None:
    ntrk = event_multiplicity(tracks)
    plt.figure()
    plt.hist(ntrk, bins=bins)
    plt.xlabel("N tracks / event within cuts")
    plt.ylabel("Events")
    plt.title("Event multiplicity distribution")
    plt.savefig(os.path.join(out_dir, "event_multiplicity.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_average_multiplicity(centrality_df: pd.DataFrame, out_dir: str) -> None:
    centrality_only = centrality_df[centrality_df["centrality"] != "all"]
    if centrality_only.empty:
        return

    plt.figure()
    plt.errorbar(
        centrality_only["centrality_mid"],
        centrality_only["mean_multiplicity"],
        yerr=centrality_only["std_multiplicity"],
        fmt="o-",
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Centrality percentile")
    plt.ylabel("Average track multiplicity")
    plt.title("Average multiplicity vs centrality")
    plt.savefig(os.path.join(out_dir, "multiplicity_vs_centrality.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_pair_correlation(
    delta_eta: np.ndarray,
    delta_phi: np.ndarray,
    out_dir: str,
    eta_bins: int,
    phi_bins: int,
) -> Optional[pd.DataFrame]:
    if len(delta_eta) == 0:
        return None

    eta_edges = np.linspace(delta_eta.min(), delta_eta.max(), eta_bins + 1)
    phi_edges = np.linspace(-math.pi, math.pi, phi_bins + 1)
    hist, eta_edges, phi_edges = np.histogram2d(delta_eta, delta_phi, bins=[eta_edges, phi_edges])

    plt.figure()
    plt.pcolormesh(eta_edges, phi_edges, hist.T, shading="auto")
    plt.xlabel(r"$\Delta\eta$")
    plt.ylabel(r"$\Delta\phi$")
    plt.title("Same-event two-particle angular correlation")
    plt.colorbar(label="Pair counts")
    plt.savefig(os.path.join(out_dir, "two_particle_correlation.png"), dpi=200, bbox_inches="tight")
    plt.close()

    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    rows = []
    for i, eta_center in enumerate(eta_centers):
        for j, phi_center in enumerate(phi_centers):
            rows.append(
                {
                    "delta_eta_center": eta_center,
                    "delta_phi_center": phi_center,
                    "pair_counts": hist[i, j],
                }
            )
    return pd.DataFrame(rows)


def build_observable_tables(
    tracks: ak.Array,
    cent: Optional[ak.Array],
    cuts: TrackCuts,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_events = len(tracks)
    multiplicity = event_multiplicity(tracks)
    cent_values = centrality_to_numpy(cent, n_events)

    eta_edges = np.linspace(-cuts.eta_abs_max, cuts.eta_abs_max, config.eta_bins + 1)
    all_pt = flatten_field(tracks, "pt")
    pt_min = max(cuts.pt_min, float(np.nanmin(all_pt)) if len(all_pt) else cuts.pt_min, 1e-3)
    pt_max = max(pt_min * 1.01, float(np.nanmax(all_pt)) if len(all_pt) else cuts.pt_max)
    pt_edges = np.logspace(np.log10(pt_min), np.log10(pt_max), config.pt_bins + 1)
    pt_edges[-1] = np.nextafter(pt_max, np.inf)

    dn_deta_tables = [compute_dn_deta(tracks, eta_edges, "all")]
    pt_tables = [compute_pt_spectrum(tracks, pt_edges, cuts.eta_abs_max, "all")]

    all_v2 = v2_two_particle(tracks)
    centrality_rows = [
        {
            "centrality": "all",
            "centrality_low": np.nan,
            "centrality_high": np.nan,
            "centrality_mid": np.nan,
            "events": n_events,
            "total_tracks": int(multiplicity.sum()),
            "mean_multiplicity": float(np.mean(multiplicity)) if n_events else np.nan,
            "std_multiplicity": float(np.std(multiplicity)) if n_events else np.nan,
            "c2": all_v2["c2"],
            "v2_2": all_v2["v2_2"],
            "v2_events_used": all_v2["events_used"],
            "v2_pairs_used": all_v2["pairs_used"],
        }
    ]

    if cent_values is not None:
        for label, lo, hi in CENTRALITY_BINS:
            mask = centrality_mask(cent_values, lo, hi)
            selected = tracks[mask]
            selected_mult = multiplicity[mask]
            if len(selected) == 0:
                continue

            dn_deta_tables.append(compute_dn_deta(selected, eta_edges, label))
            pt_tables.append(compute_pt_spectrum(selected, pt_edges, cuts.eta_abs_max, label))
            v2_result = v2_two_particle(tracks, mask)
            centrality_rows.append(
                {
                    "centrality": label,
                    "centrality_low": lo,
                    "centrality_high": hi,
                    "centrality_mid": 0.5 * (lo + hi),
                    "events": len(selected),
                    "total_tracks": int(selected_mult.sum()),
                    "mean_multiplicity": float(np.mean(selected_mult)),
                    "std_multiplicity": float(np.std(selected_mult)),
                    "c2": v2_result["c2"],
                    "v2_2": v2_result["v2_2"],
                    "v2_events_used": v2_result["events_used"],
                    "v2_pairs_used": v2_result["pairs_used"],
                }
            )

    summary = pd.DataFrame(
        [
            {
                "events": n_events,
                "total_tracks_after_cuts": int(multiplicity.sum()),
                "mean_tracks_per_event": float(np.mean(multiplicity)) if n_events else np.nan,
                "pt_min": cuts.pt_min,
                "pt_max": cuts.pt_max,
                "eta_abs_max": cuts.eta_abs_max,
                "vertex_z_abs_max": cuts.vertex_z_abs_max,
                "require_collision_flag16": cuts.require_collision_flag16,
                "event_cuts_mask": cuts.event_cuts_mask,
                "require_tpc_quality": cuts.require_tpc_quality,
                "tpc_min_found": cuts.tpc_min_found,
                "tpc_min_crossed_rows": cuts.tpc_min_crossed_rows,
                "tpc_max_chi2": cuts.tpc_max_chi2,
                "centrality_present": cent_values is not None,
                "phi_present": has_field(tracks, "phi"),
                "v2_2_all": all_v2["v2_2"],
                "c2_all": all_v2["c2"],
                "v2_pairs_used_all": all_v2["pairs_used"],
            }
        ]
    )

    return (
        pd.concat(dn_deta_tables, ignore_index=True),
        pd.concat(pt_tables, ignore_index=True),
        pd.DataFrame(centrality_rows),
        summary,
    )


def validation_report(
    tracks: ak.Array,
    cent: Optional[ak.Array],
    centrality_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    reference_comparison_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return machine-readable validation/QA checks for scientific traceability."""
    n_events = len(tracks)
    multiplicity = event_multiplicity(tracks)
    cent_values = centrality_to_numpy(cent, n_events)
    summary = summary_df.iloc[0]
    rows = []

    def add_check(name: str, status: str, detail: str) -> None:
        rows.append({"check": name, "status": status, "detail": detail})

    add_check(
        "dataset_provenance",
        "info",
        (
            f"{DATASET_PROVENANCE['record']}; DOI {DATASET_PROVENANCE['doi']}; "
            f"{DATASET_PROVENANCE['system']} at {DATASET_PROVENANCE['energy']}; "
            f"run {DATASET_PROVENANCE['run']} ({DATASET_PROVENANCE['period']})."
        ),
    )
    add_check(
        "observable_scope",
        "warning",
        (
            "Outputs are raw reconstructed-track observables. They are not corrected "
            "for detector efficiency, acceptance, secondary contamination, or systematic uncertainties."
        ),
    )
    add_check(
        "ao2d_branch_mapping",
        "warning",
        (
            "AO2D pT is derived from abs(1/fSigned1Pt), eta from asinh(fTgl), "
            "and phi from fAlpha + asin(fSnp). Validate these mappings against the "
            "O2 table definition for the exact production before final physics claims."
        ),
    )

    if n_events < 1000:
        add_check(
            "event_count",
            "warning",
            f"Only {n_events} events were analyzed; use the largest practical event sample for stable physics comparisons.",
        )
    else:
        add_check("event_count", "pass", f"{n_events} events were analyzed.")

    if cent_values is None:
        add_check("centrality_presence", "fail", "No event-level centrality values were available.")
    else:
        finite_fraction = float(np.isfinite(cent_values).mean()) if n_events else 0.0
        status = "pass" if finite_fraction >= 0.95 else "warning"
        add_check(
            "centrality_presence",
            status,
            f"{finite_fraction:.1%} of analyzed events have finite centrality values.",
        )

        cent_only = centrality_df[centrality_df["centrality"] != "all"].copy()
        if len(cent_only) >= 2:
            ordered = cent_only.sort_values("centrality_mid")
            means = ordered["mean_multiplicity"].to_numpy(dtype=float)
            monotonic = bool(np.all(np.diff(means) <= 0))
            add_check(
                "centrality_multiplicity_ordering",
                "pass" if monotonic else "warning",
                (
                    "Mean multiplicity decreases from central to peripheral bins."
                    if monotonic
                    else "Mean multiplicity is not monotonic across centrality bins; inspect centrality mapping, event selection, and sample size."
                ),
            )

    if len(multiplicity) and np.nanmax(multiplicity) > 10.0 * max(np.nanmedian(multiplicity), 1.0):
        add_check(
            "multiplicity_outliers",
            "warning",
            (
                f"Maximum event multiplicity {int(np.nanmax(multiplicity))} is more than 10x "
                f"the median {float(np.nanmedian(multiplicity)):.1f}; inspect event/track selection."
            ),
        )
    else:
        add_check("multiplicity_outliers", "pass", "No extreme multiplicity outlier was detected by the 10x median rule.")

    if bool(summary["phi_present"]):
        v2_events = float(centrality_df.loc[centrality_df["centrality"] == "all", "v2_events_used"].iloc[0])
        coverage = v2_events / max(n_events, 1)
        status = "pass" if coverage >= 0.8 else "warning"
        add_check(
            "v2_event_coverage",
            status,
            f"v2 proxy used {int(v2_events)} of {n_events} events ({coverage:.1%}); events with fewer than two finite phi tracks are skipped.",
        )
        add_check(
            "v2_physics_status",
            "warning",
            "v2{2} is a raw two-particle proxy without nonflow subtraction, eta gaps, weights, corrections, or systematic uncertainties.",
        )
    else:
        add_check("v2_event_coverage", "fail", "No phi values were available, so v2 and pair correlations cannot be validated.")

    if reference_comparison_df is None or reference_comparison_df.empty:
        add_check(
            "reference_comparison",
            "fail",
            (
                "No official/reference ALICE comparison table was evaluated. Compare dN/deta, pT spectra, "
                "centrality QA, and flow observables before claiming scientific validation."
            ),
        )
    else:
        counts = reference_comparison_df["status"].value_counts().to_dict()
        failed = int(counts.get("fail", 0))
        warnings = int(counts.get("warning", 0))
        passed = int(counts.get("pass", 0))
        status = "pass" if failed == 0 and warnings == 0 else "warning"
        if failed:
            status = "fail"
        add_check(
            "reference_comparison",
            status,
            (
                f"Official ALICE dN/deta reference comparisons: {passed} pass, "
                f"{warnings} warning, {failed} fail. Current comparison covers only raw mid-rapidity dN/deta."
            ),
        )
    return pd.DataFrame(rows)


def compare_dndeta_to_reference(
    dn_deta_df: pd.DataFrame,
    reference_path: str = REFERENCE_DNDETA_PATH,
) -> pd.DataFrame:
    """
    Compare project centrality-binned dN/deta to official ALICE mid-rapidity references.

    The reference table is for primary charged-particle dNch/deta in |eta| < 0.5.
    The project output is a raw reconstructed-track density, so this comparison is
    a conservative QA check and not a correction procedure.
    """
    if not os.path.exists(reference_path):
        return pd.DataFrame()

    references = pd.read_csv(reference_path)
    rows = []
    for ref in references.to_dict("records"):
        centrality = ref["centrality"]
        eta_abs_max = float(ref["eta_abs_max"])
        selected = dn_deta_df[
            (dn_deta_df["centrality"] == centrality)
            & (dn_deta_df["bin_center"] >= -eta_abs_max)
            & (dn_deta_df["bin_center"] <= eta_abs_max)
        ]
        if selected.empty:
            rows.append(
                {
                    **ref,
                    "measured_value": np.nan,
                    "absolute_difference": np.nan,
                    "relative_difference": np.nan,
                    "sigma_difference": np.nan,
                    "status": "fail",
                    "detail": f"No project dN/deta rows found for centrality {centrality}.",
                }
            )
            continue

        measured = float(selected["dN_deta"].mean())
        expected = float(ref["value"])
        uncertainty = float(ref["uncertainty_total"])
        absolute_difference = measured - expected
        relative_difference = absolute_difference / expected if expected else np.nan
        sigma_difference = absolute_difference / uncertainty if uncertainty else np.nan
        passes_relative = abs(relative_difference) <= float(ref["tolerance_relative"])
        passes_sigma = abs(sigma_difference) <= float(ref["tolerance_sigma"])
        status = "pass" if passes_relative or passes_sigma else "fail"
        if status == "pass" and abs(relative_difference) > 0.15:
            status = "warning"

        rows.append(
            {
                **ref,
                "measured_value": measured,
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "sigma_difference": sigma_difference,
                "status": status,
                "detail": (
                    f"Compared mean raw project dN/deta in |eta| < {eta_abs_max} "
                    f"to ALICE primary charged-particle reference."
                ),
            }
        )

    return pd.DataFrame(rows)


def run_analysis(
    tracks: ak.Array,
    cent: Optional[ak.Array],
    out_dir: str,
    cuts: TrackCuts,
    config: AnalysisConfig,
) -> Dict[str, str]:
    ensure_out_dir(out_dir)

    dn_deta_df, pt_df, centrality_df, summary_df = build_observable_tables(
        tracks, cent, cuts, config
    )

    output_paths = {
        "summary_csv": os.path.join(out_dir, "summary.csv"),
        "dn_deta_csv": os.path.join(out_dir, "dn_deta.csv"),
        "pt_spectrum_csv": os.path.join(out_dir, "pt_spectrum.csv"),
        "centrality_summary_csv": os.path.join(out_dir, "centrality_summary.csv"),
        "validation_report_csv": os.path.join(out_dir, "validation_report.csv"),
    }
    reference_comparison_df = compare_dndeta_to_reference(dn_deta_df)
    output_paths["reference_comparison_csv"] = os.path.join(out_dir, "reference_comparison.csv")
    validation_df = validation_report(
        tracks,
        cent,
        centrality_df,
        summary_df,
        reference_comparison_df,
    )
    summary_df.to_csv(output_paths["summary_csv"], index=False)
    dn_deta_df.to_csv(output_paths["dn_deta_csv"], index=False)
    pt_df.to_csv(output_paths["pt_spectrum_csv"], index=False)
    centrality_df.to_csv(output_paths["centrality_summary_csv"], index=False)
    validation_df.to_csv(output_paths["validation_report_csv"], index=False)
    reference_comparison_df.to_csv(output_paths["reference_comparison_csv"], index=False)

    plot_dn_deta_table(
        dn_deta_df[dn_deta_df["centrality"] == "all"],
        out_dir,
        "eta_distribution.png",
    )
    plot_pt_spectrum_table(
        pt_df[pt_df["centrality"] == "all"],
        out_dir,
        "pt_spectrum.png",
    )
    plot_multiplicity(tracks, out_dir)

    if len(centrality_df[centrality_df["centrality"] != "all"]) > 0:
        plot_dn_deta_table(dn_deta_df, out_dir, "dn_deta_by_centrality.png")
        plot_pt_spectrum_table(pt_df, out_dir, "pt_by_centrality.png")
        plot_average_multiplicity(centrality_df, out_dir)

    delta_eta, delta_phi = sampled_pair_deltas(
        tracks,
        max_events=config.max_pair_events,
        max_tracks=config.max_pair_tracks,
        seed=config.random_seed,
    )
    pair_df = plot_pair_correlation(
        delta_eta,
        delta_phi,
        out_dir,
        eta_bins=config.pair_eta_bins,
        phi_bins=config.pair_phi_bins,
    )
    if pair_df is not None:
        pair_path = os.path.join(out_dir, "two_particle_correlation.csv")
        pair_df.to_csv(pair_path, index=False)
        output_paths["two_particle_correlation_csv"] = pair_path

    return output_paths


# -------------------------
# CLI
# -------------------------

def inspect_first_file(data_dir: str, pattern: str) -> None:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {data_dir} with pattern {pattern}")

    info = list_trees_and_branches(files[0], max_items=40)
    print(f"\nINSPECT: {files[0]}\n")
    for tree_name, branches in info.items():
        print(f"Tree: {tree_name}\n  branches (first 40): {branches}\n")
    print("Tip: pick a tree and pass --tree_name '<TreeName>' if auto-guess fails.")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing ALICE ROOT files")
    parser.add_argument("--pattern", default="*.root", help="Glob pattern for ROOT files")
    parser.add_argument("--max_files", type=int, default=5, help="How many ROOT files to process")
    parser.add_argument("--tree_name", default=None, help="Explicit TTree name")
    parser.add_argument(
        "--file_format",
        choices=["auto", "ao2d", "aliesd", "generic"],
        default="auto",
        help="Input format. Use aliesd for classic AliESDs.root in an AliRoot/PyROOT environment.",
    )
    parser.add_argument("--max_events_per_file", type=int, default=None, help="Limit entries per file")
    parser.add_argument("--out_dir", default="outputs", help="Where to save plots and CSV tables")

    parser.add_argument("--pt_min", type=float, default=0.15)
    parser.add_argument("--pt_max", type=float, default=50.0)
    parser.add_argument("--eta_abs_max", type=float, default=0.8)
    parser.add_argument("--vertex_z_abs_max", type=float, default=10.0)
    parser.add_argument("--disable_collision_flag16", action="store_true")
    parser.add_argument("--event_cuts_mask", type=int, default=1023)
    parser.add_argument("--disable_tpc_quality", action="store_true")
    parser.add_argument("--tpc_min_found", type=int, default=70)
    parser.add_argument("--tpc_min_crossed_rows", type=int, default=70)
    parser.add_argument("--tpc_max_chi2", type=float, default=4.0)

    parser.add_argument("--eta_bins", type=int, default=60)
    parser.add_argument("--pt_bins", type=int, default=80)
    parser.add_argument("--max_pair_events", type=int, default=500)
    parser.add_argument("--max_pair_tracks", type=int, default=250)
    parser.add_argument("--inspect", action="store_true", help="Print trees/branches and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.inspect:
        inspect_first_file(args.data_dir, args.pattern)
        return

    cuts = TrackCuts(
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        eta_abs_max=args.eta_abs_max,
        vertex_z_abs_max=args.vertex_z_abs_max,
        require_collision_flag16=not args.disable_collision_flag16,
        event_cuts_mask=args.event_cuts_mask,
        require_tpc_quality=not args.disable_tpc_quality,
        tpc_min_found=args.tpc_min_found,
        tpc_min_crossed_rows=args.tpc_min_crossed_rows,
        tpc_max_chi2=args.tpc_max_chi2,
    )
    config = AnalysisConfig(
        eta_bins=args.eta_bins,
        pt_bins=args.pt_bins,
        max_pair_events=args.max_pair_events,
        max_pair_tracks=args.max_pair_tracks,
    )

    tracks, cent = load_dataset(
        data_dir=args.data_dir,
        pattern=args.pattern,
        max_files=args.max_files,
        tree_name=args.tree_name,
        cuts=cuts,
        max_events_per_file=args.max_events_per_file,
        file_format=args.file_format,
    )

    output_paths = run_analysis(tracks, cent, args.out_dir, cuts, config)
    summary = pd.read_csv(output_paths["summary_csv"]).iloc[0]

    print("\nLoaded dataset summary:")
    print(f"- events: {int(summary['events'])}")
    print(f"- total tracks after cuts: {int(summary['total_tracks_after_cuts'])}")
    print(f"- centrality present: {bool(summary['centrality_present'])}")
    print(f"- phi present: {bool(summary['phi_present'])}")
    print(f"- raw v2{{2}} proxy: {summary['v2_2_all']}")

    print(f"\nSaved analysis outputs to: {args.out_dir}/")
    for name in sorted(output_paths):
        print(f"  - {os.path.basename(output_paths[name])}")
    print("  - eta_distribution.png")
    print("  - pt_spectrum.png")
    print("  - event_multiplicity.png")
    if bool(summary["centrality_present"]):
        print("  - dn_deta_by_centrality.png")
        print("  - pt_by_centrality.png")
        print("  - multiplicity_vs_centrality.png")
    if bool(summary["phi_present"]):
        print("  - two_particle_correlation.png")


if __name__ == "__main__":
    main()
