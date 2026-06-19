"""
Streamlit dashboard for the ALICE heavy-ion analysis pipeline.

Run:
  streamlit run streamlit_dashboard.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from heavy_ion_alice_analysis import (
    AnalysisConfig,
    TrackCuts,
    inspect_first_file,
    load_dataset,
    run_analysis,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_ROOT_FILE = APP_DIR / "sample_data" / "AO2D_sample.root"


def prepare_data_source(source_mode, uploaded_files, server_data_dir, server_pattern):
    """Return (data directory, glob pattern, temporary directory handle)."""
    if source_mode == "Bundled sample":
        if not SAMPLE_ROOT_FILE.exists():
            raise FileNotFoundError("The bundled sample ROOT file is missing from this deployment.")
        return str(SAMPLE_ROOT_FILE.parent), SAMPLE_ROOT_FILE.name, None

    if source_mode == "Upload ROOT files":
        if not uploaded_files:
            raise ValueError("Upload at least one .root file first.")

        temp_dir = tempfile.TemporaryDirectory(prefix="heavy-ion-upload-")
        for index, uploaded_file in enumerate(uploaded_files):
            safe_name = Path(uploaded_file.name).name
            destination = Path(temp_dir.name) / f"{index:03d}_{safe_name}"
            uploaded_file.seek(0)
            with destination.open("wb") as output_file:
                shutil.copyfileobj(uploaded_file, output_file)
        return temp_dir.name, "*.root", temp_dir

    if not server_data_dir:
        raise ValueError("Enter a ROOT file folder first.")
    return server_data_dir, server_pattern, None


st.set_page_config(page_title="ALICE Heavy-Ion Analysis", layout="wide")

st.title("ALICE Heavy-Ion Collision Analysis")

with st.sidebar:
    st.header("Input")
    source_mode = st.radio(
        "Data source",
        ["Bundled sample", "Upload ROOT files", "Server folder"],
        help="Try the included AO2D sample, upload your own ROOT files, or use a folder available on the server.",
    )
    uploaded_files = None
    data_dir = ""
    pattern = "*.root"
    if source_mode == "Bundled sample":
        sample_size_mb = SAMPLE_ROOT_FILE.stat().st_size / (1024 * 1024) if SAMPLE_ROOT_FILE.exists() else 0
        st.caption(f"Included AO2D sample ({sample_size_mb:.1f} MB)")
    elif source_mode == "Upload ROOT files":
        uploaded_files = st.file_uploader(
            "ROOT files",
            type=["root"],
            accept_multiple_files=True,
            help="Files are copied to temporary server storage only for this analysis run.",
        )
    else:
        data_dir = st.text_input("ROOT file folder", value="")
        pattern = st.text_input("File pattern", value="*.root")
    file_format = st.selectbox(
        "File format",
        ["auto", "ao2d", "aliesd", "generic"],
        help="Use aliesd for classic AliESDs.root inside an AliRoot/PyROOT environment.",
    )
    tree_name = st.text_input("Tree name", value="")
    max_files = st.number_input("Number of ROOT files", min_value=1, max_value=200, value=5)
    max_events_per_file = st.number_input(
        "Max events per file",
        min_value=0,
        max_value=1_000_000,
        value=0,
        help="Use 0 for no limit.",
    )

    st.header("Track cuts")
    pt_min = st.number_input("pT min [GeV/c]", min_value=0.0, value=0.15, step=0.05)
    pt_max = st.number_input("pT max [GeV/c]", min_value=0.1, value=50.0, step=1.0)
    eta_abs_max = st.number_input("|eta| max", min_value=0.1, value=0.8, step=0.1)
    vertex_z_abs_max = st.number_input("|vertex z| max [cm]", min_value=1.0, value=10.0, step=1.0)
    require_collision_flag16 = st.checkbox("Require selected collision flag", value=True)
    event_cuts_mask = st.number_input("Event cuts bit mask", min_value=0, value=1023, step=1)
    require_tpc_quality = st.checkbox("Apply TPC track-quality cuts", value=True)
    tpc_min_found = st.number_input("TPC min found clusters", min_value=0, value=70, step=5)
    tpc_min_crossed_rows = st.number_input("TPC min crossed rows", min_value=0, value=70, step=5)
    tpc_max_chi2 = st.number_input("TPC max chi2/cluster", min_value=0.1, value=4.0, step=0.5)

    st.header("Analysis controls")
    eta_bins = st.slider("eta bins", min_value=20, max_value=160, value=60)
    pt_bins = st.slider("pT bins", min_value=20, max_value=160, value=80)
    max_pair_events = st.slider("Max events for pair correlations", 10, 5000, 500)
    max_pair_tracks = st.slider("Max tracks/event for pair correlations", 20, 1000, 250)
    out_dir = st.text_input("Output folder", value="outputs/dashboard")

    inspect_clicked = st.button("Inspect first file")
    run_clicked = st.button("Run analysis", type="primary")


if inspect_clicked:
    temp_dir = None
    try:
        resolved_data_dir, resolved_pattern, temp_dir = prepare_data_source(
            source_mode, uploaded_files, data_dir, pattern
        )
        try:
            with st.spinner("Inspecting ROOT file..."):
                import io
                import contextlib

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    inspect_first_file(resolved_data_dir, resolved_pattern)
                st.code(buffer.getvalue(), language="text")
        except Exception as exc:
            st.error(str(exc))
    except Exception as exc:
        st.error(str(exc))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if run_clicked:
    if pt_max <= pt_min:
        st.error("pT max must be larger than pT min.")
    else:
        cuts = TrackCuts(
            pt_min=pt_min,
            pt_max=pt_max,
            eta_abs_max=eta_abs_max,
            vertex_z_abs_max=vertex_z_abs_max,
            require_collision_flag16=require_collision_flag16,
            event_cuts_mask=int(event_cuts_mask),
            require_tpc_quality=require_tpc_quality,
            tpc_min_found=int(tpc_min_found),
            tpc_min_crossed_rows=int(tpc_min_crossed_rows),
            tpc_max_chi2=tpc_max_chi2,
        )
        config = AnalysisConfig(
            eta_bins=eta_bins,
            pt_bins=pt_bins,
            max_pair_events=max_pair_events,
            max_pair_tracks=max_pair_tracks,
        )

        temp_dir = None
        try:
            resolved_data_dir, resolved_pattern, temp_dir = prepare_data_source(
                source_mode, uploaded_files, data_dir, pattern
            )
            with st.spinner("Loading ROOT files and running analysis..."):
                tracks, cent = load_dataset(
                    data_dir=resolved_data_dir,
                    pattern=resolved_pattern,
                    max_files=int(max_files),
                    tree_name=tree_name or None,
                    cuts=cuts,
                    max_events_per_file=int(max_events_per_file) or None,
                    file_format=file_format,
                )
                output_paths = run_analysis(tracks, cent, out_dir, cuts, config)

            st.success(f"Analysis complete. Outputs saved in {out_dir}.")

            summary = pd.read_csv(output_paths["summary_csv"])
            centrality = pd.read_csv(output_paths["centrality_summary_csv"])
            dn_deta = pd.read_csv(output_paths["dn_deta_csv"])
            pt_spectrum = pd.read_csv(output_paths["pt_spectrum_csv"])
            validation = pd.read_csv(output_paths["validation_report_csv"])
            reference_comparison = pd.read_csv(output_paths["reference_comparison_csv"])

            c1, c2, c3, c4 = st.columns(4)
            row = summary.iloc[0]
            c1.metric("Events", int(row["events"]))
            c2.metric("Tracks after cuts", int(row["total_tracks_after_cuts"]))
            c3.metric("Mean tracks/event", f"{row['mean_tracks_per_event']:.2f}")
            c4.metric("Raw v2{2} proxy", "n/a" if pd.isna(row["v2_2_all"]) else f"{row['v2_2_all']:.4f}")

            status_counts = validation["status"].value_counts().to_dict()
            if status_counts.get("fail", 0) or status_counts.get("warning", 0):
                st.warning(
                    "Validation report contains warnings or failures. Treat plots as raw exploratory outputs, not final ALICE physics results."
                )
            else:
                st.success("Basic validation checks passed for this run.")

            plot_choice = st.selectbox(
                "Plot",
                [
                    "dN/deta by centrality",
                    "pT spectra by centrality",
                    "Average multiplicity vs centrality",
                    "Two-particle angular correlation",
                ],
            )

            if plot_choice == "dN/deta by centrality":
                st.line_chart(
                    dn_deta,
                    x="bin_center",
                    y="dN_deta",
                    color="centrality",
                )
                image_path = os.path.join(out_dir, "dn_deta_by_centrality.png")
                if os.path.exists(image_path):
                    st.image(image_path)

            elif plot_choice == "pT spectra by centrality":
                st.line_chart(
                    pt_spectrum,
                    x="bin_center",
                    y="dN_dpT_per_event",
                    color="centrality",
                )
                image_path = os.path.join(out_dir, "pt_by_centrality.png")
                if os.path.exists(image_path):
                    st.image(image_path)

            elif plot_choice == "Average multiplicity vs centrality":
                cent_only = centrality[centrality["centrality"] != "all"]
                if cent_only.empty:
                    st.info("No centrality branch was found, so this plot is unavailable.")
                else:
                    st.line_chart(
                        cent_only,
                        x="centrality_mid",
                        y="mean_multiplicity",
                    )
                    image_path = os.path.join(out_dir, "multiplicity_vs_centrality.png")
                    if os.path.exists(image_path):
                        st.image(image_path)

            elif plot_choice == "Two-particle angular correlation":
                image_path = os.path.join(out_dir, "two_particle_correlation.png")
                if os.path.exists(image_path):
                    st.image(image_path)
                else:
                    st.info("No phi branch was found, so pair correlations are unavailable.")

            st.subheader("Summary table")
            st.dataframe(summary, width="stretch")

            st.subheader("Centrality summary")
            st.dataframe(centrality, width="stretch")

            st.subheader("Validation report")
            st.dataframe(validation, width="stretch")

            st.subheader("ALICE reference comparison")
            st.dataframe(reference_comparison, width="stretch")

            st.subheader("CSV outputs")
            for label, path in output_paths.items():
                if os.path.exists(path):
                    with open(path, "rb") as file:
                        st.download_button(
                            label=f"Download {os.path.basename(path)}",
                            data=file,
                            file_name=os.path.basename(path),
                            mime="text/csv",
                            key=label,
                        )

        except Exception as exc:
            st.error(str(exc))
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
else:
    st.info("Use the bundled sample for a quick test, or choose another data source in the sidebar.")
