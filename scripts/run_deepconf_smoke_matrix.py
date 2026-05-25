#!/usr/bin/env python3
"""
Run a DeepConf smoke-test matrix on a remote machine.

This script is intentionally self-contained: it creates a small input SDF when
one is not provided, runs selected DeepConf workflow combinations in separate
folders, and writes a summary report plus per-case logs.

Example:
    python scripts/run_deepconf_smoke_matrix.py --calculator ani2x

For a real ligand:
    python scripts/run_deepconf_smoke_matrix.py --input-sdf /path/to/ligand.sdf
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rdkit import Chem
from rdkit.Chem import AllChem


def str_bool(value):
    return "yes" if value else "no"


def create_ethanol_sdf(path):
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
    mol.SetProp("_Name", "ethanol")
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def read_sdf_summary(path):
    if not path.exists():
        return {
            "exists": False,
            "n_molecules": 0,
            "all_have_energy": False,
            "same_topology": False,
        }

    mols = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]
    if not mols:
        return {
            "exists": True,
            "n_molecules": 0,
            "all_have_energy": False,
            "same_topology": False,
        }

    first_atoms = mols[0].GetNumAtoms()
    first_bonds = mols[0].GetNumBonds()
    return {
        "exists": True,
        "n_molecules": len(mols),
        "all_have_energy": all(mol.HasProp("Energy") for mol in mols),
        "same_topology": all(
            mol.GetNumAtoms() == first_atoms and mol.GetNumBonds() == first_bonds
            for mol in mols
        ),
        "atom_count": first_atoms,
        "bond_count": first_bonds,
    }


def expected_output(case_dir, file_base, case, add_hydrogen=False):
    if not case["verbose"]:
        return case_dir / f"{file_base}_output.sdf"

    work_dir = case_dir / file_base
    if not case["genconformer"]:
        prefix = ""
        if add_hydrogen:
            prefix += "addH_"
        if (
            case["optimization_lig"]
            or case["optimization_conf"]
            or case["pre_optimization_lig"]
        ):
            prefix += "opt_"
        return work_dir / f"global_{prefix}{file_base}.sdf"

    if case["optimization_conf"]:
        return work_dir / "opt_picked_confs" / f"{file_base}_output.sdf"

    return work_dir / "picked_confs" / f"{file_base}_output.sdf"


def build_cases(full=False):
    core_cases = [
        {
            "name": "geom_only_verbose_yes",
            "genconformer": False,
            "pre_optimization_lig": False,
            "optimization_conf": False,
            "optimization_lig": True,
            "verbose": True,
            "organize_clusters": True,
            "organize_mode": "move",
        },
        {
            "name": "preopt_confs_no_confopt_verbose_yes",
            "genconformer": True,
            "pre_optimization_lig": True,
            "optimization_conf": False,
            "optimization_lig": False,
            "verbose": True,
            "organize_clusters": True,
            "organize_mode": "move",
        },
        {
            "name": "preopt_confs_confopt_verbose_yes",
            "genconformer": True,
            "pre_optimization_lig": True,
            "optimization_conf": True,
            "optimization_lig": False,
            "verbose": True,
            "organize_clusters": True,
            "organize_mode": "move",
        },
        {
            "name": "confs_no_confopt_verbose_yes",
            "genconformer": True,
            "pre_optimization_lig": False,
            "optimization_conf": False,
            "optimization_lig": False,
            "verbose": True,
            "organize_clusters": True,
            "organize_mode": "move",
        },
        {
            "name": "confs_confopt_verbose_yes",
            "genconformer": True,
            "pre_optimization_lig": False,
            "optimization_conf": True,
            "optimization_lig": False,
            "verbose": True,
            "organize_clusters": True,
            "organize_mode": "move",
        },
        {
            "name": "geom_only_verbose_no",
            "genconformer": False,
            "pre_optimization_lig": False,
            "optimization_conf": False,
            "optimization_lig": True,
            "verbose": False,
            "organize_clusters": True,
            "organize_mode": "move",
        },
        {
            "name": "confs_confopt_verbose_no",
            "genconformer": True,
            "pre_optimization_lig": False,
            "optimization_conf": True,
            "optimization_lig": False,
            "verbose": False,
            "organize_clusters": True,
            "organize_mode": "move",
        },
    ]

    if not full:
        return core_cases

    extra_cases = []
    for organize_clusters in (False, True):
        for organize_mode in ("move", "copy"):
            extra_cases.append({
                "name": f"confs_confopt_org_{str_bool(organize_clusters)}_{organize_mode}",
                "genconformer": True,
                "pre_optimization_lig": False,
                "optimization_conf": True,
                "optimization_lig": False,
                "verbose": True,
                "organize_clusters": organize_clusters,
                "organize_mode": organize_mode,
            })
            extra_cases.append({
                "name": f"preopt_confs_confopt_org_{str_bool(organize_clusters)}_{organize_mode}",
                "genconformer": True,
                "pre_optimization_lig": True,
                "optimization_conf": True,
                "optimization_lig": False,
                "verbose": True,
                "organize_clusters": organize_clusters,
                "organize_mode": organize_mode,
            })

    return core_cases + extra_cases


def build_command(args, input_dir, case):
    return [
        str(args.python),
        str(args.repo_root / "runConfGen.py"),
        str(input_dir),
        str_bool(args.add_hydrogen),
        args.calculator,
        args.optimization_method,
        str_bool(case["optimization_conf"]),
        str_bool(case["optimization_lig"]),
        str_bool(case["pre_optimization_lig"]),
        str_bool(case["genconformer"]),
        str(args.nprocs),
        str(args.thr_fmax),
        str(args.maxiter),
        str_bool(args.etkdg),
        str(args.num_conformers),
        str(args.max_attempts),
        str(args.prune_rms_thresh),
        str(args.opt_prune_rms_thresh),
        str(args.opt_prune_diffE_thresh),
        str(args.nfold),
        str(args.npick),
        str(args.nscale),
        str(args.cluster_nprocs),
        str(args.cluster_chunk_size),
        args.cluster_linkage,
        str_bool(case["organize_clusters"]),
        case["organize_mode"],
        args.summary_csv,
        str_bool(case["verbose"]),
        args.calculator_model,
        str(args.calculator_charge),
        str(args.calculator_multiplicity),
        args.calculator_device,
        args.nequip_chemical_symbols,
    ]


def run_case(args, case, source_sdf, file_base, output_root):
    case_dir = output_root / case["name"]
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sdf, input_dir / f"{file_base}.sdf")

    cmd = build_command(args, input_dir, case)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(case_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    elapsed = time.time() - start

    log_path = case_dir / "run.log"
    log_path.write_text(proc.stdout)

    expected_sdf = expected_output(case_dir, file_base, case, add_hydrogen=args.add_hydrogen)
    sdf_summary = read_sdf_summary(expected_sdf)
    work_dir = case_dir / file_base

    passed = (
        proc.returncode == 0
        and sdf_summary["exists"]
        and sdf_summary["n_molecules"] > 0
        and sdf_summary["all_have_energy"]
        and sdf_summary["same_topology"]
    )

    if not case["verbose"] and work_dir.exists():
        passed = False

    return {
        "name": case["name"],
        "passed": passed,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "expected_sdf": str(expected_sdf),
        "log": str(log_path),
        "command": " ".join(cmd),
        "case": case,
        "sdf_summary": sdf_summary,
        "work_dir_exists": work_dir.exists(),
    }


def write_markdown_report(path, results):
    lines = [
        "# DeepConf Smoke Matrix Report",
        "",
        "| Case | Pass | Return code | Molecules | Energy props | Same topology | Log |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        sdf_summary = result["sdf_summary"]
        lines.append(
            "| {name} | {passed} | {returncode} | {n_molecules} | {all_have_energy} | "
            "{same_topology} | {log} |".format(
                name=result["name"],
                passed="yes" if result["passed"] else "no",
                returncode=result["returncode"],
                n_molecules=sdf_summary.get("n_molecules", 0),
                all_have_energy=sdf_summary.get("all_have_energy", False),
                same_topology=sdf_summary.get("same_topology", False),
                log=result["log"],
            )
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--input-sdf", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--calculator", default="ani2x")
    parser.add_argument("--calculator-model", default="")
    parser.add_argument("--calculator-charge", default="")
    parser.add_argument("--calculator-multiplicity", type=int, default=1)
    parser.add_argument("--calculator-device", default="auto")
    parser.add_argument("--nequip-chemical-symbols", default="")
    parser.add_argument("--optimization-method", default="BFGS")
    parser.add_argument("--add-hydrogen", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--case-filter", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=900)

    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--thr-fmax", type=float, default=10.0)
    parser.add_argument("--maxiter", type=int, default=1)
    parser.add_argument("--no-etkdg", dest="etkdg", action="store_false")
    parser.set_defaults(etkdg=True)
    parser.add_argument("--num-conformers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--opt-prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--opt-prune-diffE-thresh", type=float, default=0.001)
    parser.add_argument("--nfold", type=int, default=1)
    parser.add_argument("--npick", type=int, default=0)
    parser.add_argument("--nscale", type=int, default=1)
    parser.add_argument("--cluster-nprocs", type=int, default=1)
    parser.add_argument("--cluster-chunk-size", type=int, default=50)
    parser.add_argument("--cluster-linkage", default="complete")
    parser.add_argument("--summary-csv", default="cluster_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()

    if args.output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_root = args.repo_root / "smoke_runs" / stamp
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.input_sdf is None:
        source_sdf = args.output_root / "ethanol.sdf"
        create_ethanol_sdf(source_sdf)
        file_base = "ethanol"
    else:
        source_sdf = args.input_sdf.resolve()
        file_base = source_sdf.stem

    results = []
    cases = build_cases(full=args.full)
    if args.case_filter:
        cases = [
            case for case in cases
            if any(case_filter in case["name"] for case_filter in args.case_filter)
        ]

    if not cases:
        raise ValueError("No smoke-test cases matched the requested filters")

    for case in cases:
        print(f"Running {case['name']} ...", flush=True)
        try:
            results.append(run_case(args, case, source_sdf, file_base, args.output_root))
        except subprocess.TimeoutExpired as exc:
            results.append({
                "name": case["name"],
                "passed": False,
                "returncode": "timeout",
                "elapsed_s": args.timeout,
                "expected_sdf": "",
                "log": "",
                "case": case,
                "sdf_summary": {
                    "exists": False,
                    "n_molecules": 0,
                    "all_have_energy": False,
                    "same_topology": False,
                },
                "error": str(exc),
            })

    report_json = args.output_root / "smoke_report.json"
    report_md = args.output_root / "smoke_report.md"
    report_json.write_text(json.dumps(results, indent=2))
    write_markdown_report(report_md, results)

    n_passed = sum(1 for result in results if result["passed"])
    print(f"\nPassed {n_passed}/{len(results)} cases")
    print(f"Report JSON: {report_json}")
    print(f"Report Markdown: {report_md}")

    if n_passed != len(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
