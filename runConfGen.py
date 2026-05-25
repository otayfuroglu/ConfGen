
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np

from ase.io import read, write
from core_conf import confGen
import argparse
import os, sys, shutil
import multiprocessing
from itertools import product
import time
import traceback

nprocs_all = int(multiprocessing.cpu_count())



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("structure_dir", type=str)
parser.add_argument("ignore_hydrogen", nargs="?", default="No") # args for bool
parser.add_argument("calculator_type", type=str)
parser.add_argument("optimization_method", nargs="?", default="No") # args for bool
parser.add_argument("optimization_conf", nargs="?", default="No") # args for bool
parser.add_argument("optimization_lig", nargs="?", default="No") # args for bool
parser.add_argument("pre_optimization_lig", nargs="?", default="No") # args for bool
parser.add_argument("genconformer", nargs="?", default="No") # args for bool
parser.add_argument("nprocs", type=int, default=nprocs_all)
parser.add_argument("thr_fmax", type=float, default=0.05)
parser.add_argument("maxiter", type=int, default=500)

parser.add_argument("ETKDG", nargs="?", default="No") # args for bool
parser.add_argument("num_conformers", type=int, default=50)
parser.add_argument("max_attempts", type=int, default=100)
parser.add_argument("prune_rms_thresh", type=float, default=0.2)
parser.add_argument("opt_prune_rms_thresh", type=float, default=0.2)
parser.add_argument("opt_prune_diffE_thresh", type=float, default=0.001)
parser.add_argument("nfold", type=int, default=2)
parser.add_argument("npick", type=int, default=2)
parser.add_argument("nscale", type=int, default=2)

# Optional trailing arguments keep the existing positional interface compatible.
parser.add_argument("cluster_nprocs", nargs="?", type=int, default=nprocs_all)
parser.add_argument("cluster_chunk_size", nargs="?", type=int, default=4000)
parser.add_argument("cluster_linkage", nargs="?", default="complete")
parser.add_argument("organize_clusters", nargs="?", default="yes")
parser.add_argument("organize_mode", nargs="?", default="move")
parser.add_argument("summary_csv", nargs="?", default="cluster_summary.csv")
parser.add_argument("verbose", nargs="?", default="yes")
parser.add_argument("calculator_model", nargs="?", default="")
parser.add_argument("calculator_charge", nargs="?", default="")
parser.add_argument("calculator_mult", nargs="?", default="1")
parser.add_argument("calculator_device", nargs="?", default="auto")
parser.add_argument("nequip_chemical_symbols", nargs="?", default="")
parser.add_argument("g16_mem", nargs="?", default="4GB")
parser.add_argument("g16_level", nargs="?", default="WB97XD")
parser.add_argument("g16_basis", nargs="?", default="6-311++G(3df,3pd)")


def calcFuncRunTime(func):
    import time
    def wrapper(*args, **kwargs):
        s_time = time.time()
        func(*args, **kwargs)
        print(f"Funtion {func.__name__} executed in {(time.time()-s_time)/60:.4f} m")
    return wrapper


def getBoolStr(string):
    string = string.lower()
    if "true" in string or "yes" in string:
        return True
    elif "false" in string or "no" in string:
        return False
    else:
        print("%s is bad input!!! Must be Yes/No or True/False" %string)
        sys.exit(1)


def _value_or_env(value, env_name, default=""):
    if value is None or str(value).strip() == "":
        return os.environ.get(env_name, default)
    return value


def _parse_nequip_chemical_symbols(value):
    value = _value_or_env(value, "DEEPCONF_NEQUIP_CHEMICAL_SYMBOLS", "")
    value = str(value).strip()
    if value == "":
        return None
    if value.lower() in ("true", "yes", "identity"):
        return True

    try:
        import json
        return json.loads(value)
    except Exception:
        pass

    if ":" in value:
        return dict(
            item.split(":", 1)
            for item in value.split(",")
            if item.strip()
        )

    return [item.strip() for item in value.split(",") if item.strip()]


_BUNDLED_NEQUIP_MODELS = {
    "G_NequIP.pth",
    "G_NequIP_smdW.pth",
    "M_NequIP.pth",
    "M_NequIP_smdO.pth",
    "M_NequIP_smdW.pth",
}


def _default_nequip_model_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "all_NNP_MODELS",
        "G_NequIP.pth",
    )


def _is_bundled_nequip_model(model_path):
    return os.path.basename(os.path.abspath(str(model_path))) in _BUNDLED_NEQUIP_MODELS


def _resolve_nequip_model_path(value):
    model_path = _value_or_env(value, "DEEPCONF_NEQUIP_MODEL", "")
    if model_path is None or str(model_path).strip() == "":
        model_path = _default_nequip_model_path()
        if verbose:
            print(f"Using default bundled NequIP model: {model_path}")
    else:
        model_path = os.path.expandvars(os.path.expanduser(str(model_path).strip()))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NequIP model file not found: {model_path}")

    return model_path


def _resolve_nequip_chemical_symbols(value, model_path):
    chemical_symbols = _parse_nequip_chemical_symbols(value)
    if chemical_symbols is None and _is_bundled_nequip_model(model_path):
        if verbose:
            print("Using identity chemical symbol mapping for bundled NequIP model")
        return True

    return chemical_symbols


def _formal_charge_from_ligand(lig):
    mol = getattr(lig, "rw_mol", None)
    if mol is None:
        return 0

    for prop_name in (
        "Charge",
        "charge",
        "TotalCharge",
        "total_charge",
        "FormalCharge",
        "formal_charge",
    ):
        if mol.HasProp(prop_name):
            try:
                return int(float(mol.GetProp(prop_name)))
            except Exception:
                pass

    return int(Chem.GetFormalCharge(mol))


def _parse_calculator_charge(value, lig, calculator_label="calculator",
                             env_name="DEEPCONF_CALCULATOR_CHARGE"):
    value = _value_or_env(value, env_name, "")
    if value is None or str(value).strip() == "":
        charge = _formal_charge_from_ligand(lig)
        if verbose:
            print(f"Using inferred molecular charge for {calculator_label}: {charge}")
        return charge

    return int(float(value))


def _read_sdf_energy_for_export(sdf_path, default=float("inf")):
    try:
        mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False))
    except Exception:
        return default

    if mol is None or not mol.HasProp("Energy"):
        return default

    try:
        return float(mol.GetProp("Energy"))
    except Exception:
        return default


def _write_combined_sdf_from_dir(sdf_dir, out_sdf):
    sdf_files = [
        os.path.join(sdf_dir, f)
        for f in os.listdir(sdf_dir)
        if f.endswith(".sdf")
    ]
    if len(sdf_files) == 0:
        return None

    sdf_files = sorted(sdf_files, key=lambda p: _read_sdf_energy_for_export(p))

    with Chem.SDWriter(out_sdf) as writer:
        for rank, sdf_path in enumerate(sdf_files, start=1):
            mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False))
            if mol is None:
                continue
            mol.SetProp("SourceFile", os.path.basename(sdf_path))
            mol.SetProp("OutputRankByEnergy", str(rank))
            writer.write(mol)

    if os.path.exists(out_sdf) and os.path.getsize(out_sdf) > 0:
        return out_sdf

    return None


def _find_final_sdf(WORK_DIR, file_base, prefix):
    exact_names = [
        f"{file_base}_output.sdf",
        f"{prefix}output.sdf",
        "output.sdf",
    ]

    candidates = []
    for root, dirs, files in os.walk(WORK_DIR):
        for name in files:
            if name in exact_names and name.endswith(".sdf"):
                candidates.append(os.path.join(root, name))

    if candidates:
        candidates = sorted(
            candidates,
            key=lambda p: (
                0 if "picked_confs" in os.path.normpath(p).split(os.sep) else 1,
                len(os.path.normpath(p).split(os.sep)),
                p,
            )
        )
        return candidates[0]

    output_candidates = []
    for root, dirs, files in os.walk(WORK_DIR):
        for name in files:
            if name.endswith("_output.sdf"):
                output_candidates.append(os.path.join(root, name))

    if output_candidates:
        output_candidates = sorted(
            output_candidates,
            key=lambda p: (
                0 if "picked_confs" in os.path.normpath(p).split(os.sep) else 1,
                len(os.path.normpath(p).split(os.sep)),
                p,
            )
        )
        return output_candidates[0]

    global_candidate = os.path.join(WORK_DIR, f"global_{prefix}{file_base}.sdf")
    if os.path.exists(global_candidate):
        return global_candidate

    pre_candidate = os.path.join(WORK_DIR, f"pre_{prefix}{file_base}.sdf")
    if os.path.exists(pre_candidate):
        return pre_candidate

    for subdir_name in ("picked_confs", "opt_picked_confs"):
        sdf_dir = os.path.join(WORK_DIR, subdir_name)
        if os.path.isdir(sdf_dir):
            combined = os.path.join(sdf_dir, f"{file_base}_output.sdf")
            made = _write_combined_sdf_from_dir(sdf_dir, combined)
            if made is not None:
                return made

    return None


def _export_final_sdf_and_cleanup(WORK_DIR, file_base, prefix):
    final_sdf = _find_final_sdf(WORK_DIR, file_base, prefix)

    if final_sdf is None or not os.path.exists(final_sdf):
        print(f"Warning: no final SDF found for {file_base}; keeping work directory: {WORK_DIR}")
        return None

    export_sdf = os.path.abspath(f"{file_base}_output.sdf")
    if os.path.exists(export_sdf):
        os.remove(export_sdf)

    shutil.copy2(final_sdf, export_sdf)
    print(f"Compact output written to: {export_sdf}")

    for attempt in range(3):
        try:
            shutil.rmtree(WORK_DIR)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.2)
    print(f"Removed work directory because verbose=no: {WORK_DIR}")

    return export_sdf


def setG16calculator(lig, file_base, label, WORK_DIR, charge=0, mult=1,
                     mem="4GB", level="WB97XD", basis="6-311++G(3df,3pd)"):
    lig.setG16Calculator(
            label="%s/g16_%s/%s"%(WORK_DIR, label, file_base),
            chk="",
            nprocs=nprocs,
            xc=level,
            basis=basis,
            scf="XQC, maxconventionalcycles=100",
            extra="nosymm",
            charge=charge,
            mult=mult,
            mem=mem,

            )
    return lig


def setGenConformers(lig, out_file_path, mmCalculator):
    last_exc = None
    for trial in range(1, 4):
        try:
            lig.genGonformers(
                file_path=out_file_path,
                numConfs=num_conformers,
                ETKDG=ETKDG,
                maxAttempts=max_attempts,
                pruneRmsThresh=prune_rms_thresh,
                mmCalculator=mmCalculator,
                optimization_conf=optimization_conf,
                opt_prune_rms_thresh=opt_prune_rms_thresh,
                opt_prune_diffE_thresh=opt_prune_diffE_thresh,
                nfold=nfold,
                npick=npick,
                cluster_nprocs=cluster_nprocs,
                cluster_chunk_size=cluster_chunk_size,
                cluster_linkage=cluster_linkage,
                organize_clusters=organize_clusters,
                organize_mode=organize_mode,
                summary_csv=summary_csv,)
            return lig
        except Exception as exc:
            last_exc = exc
            print(f"Trial {trial} failed during conformer generation/optimization/clustering/SDF writing.")
            print(f"Error type: {type(exc).__name__}")
            print(f"Error message: {exc}")
            traceback.print_exc()

            if hasattr(lig, "increaseTrialNum"):
                lig.increaseTrialNum()

    print("All 3 attempts failed.")
    raise last_exc


#  @calcFuncRunTime
def runConfGen(file_name):
    "Starting ligand preparetion process... "
    mol_path= "%s/%s"%(structure_dir, file_name)

    file_base = file_name.split(".")[0]
    #create destination directory
    WORK_DIR = file_base
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.mkdir(WORK_DIR)

    #Flags
    # default mm calculator set to False
    mmCalculator=False
    # default adding H is False
    addH = False

    # if desire adding H by openbabel
    prefix = ""
    if ignore_hydrogen:
        addH = True
        prefix += "addH_"
    if optimization_lig or optimization_conf or pre_optimization_lig:
        prefix += "opt_"

    # initialize confGen
    lig = confGen(mol_path, addH, WORK_DIR, verbose=verbose)
    lig.setVerbose(verbose)
    lig.setOptMethod(optimization_method)
    #  lig.writeRWMol2File("test/test.xyz")

    calculator_key = calculator_type.lower().replace("-", "").replace("_", "")
    if calculator_key in ("ani1x", "ani1ccx", "ani2x"):
        lig.setANICalculator(calculator_key)
    elif "g16" in calculator_type.lower():
        lig = setG16calculator(
            lig,
            file_base,
            label="calculation",
            WORK_DIR=WORK_DIR,
            charge=_parse_calculator_charge(
                calculator_charge,
                lig,
                calculator_label="Gaussian16",
                env_name="DEEPCONF_G16_CHARGE",
            ),
            mult=calculator_mult,
            mem=g16_mem,
            level=g16_level,
            basis=g16_basis,
        )
    elif "uff" in calculator_type.lower():
        if optimization_conf:
            print("UFF calculator not support optimization")
            sys.exit(1)
        else:
            mmCalculator=True
    elif "aimnet" in calculator_type.lower():
        default_model = "aimnet2"
        if calculator_type.lower() not in ("aimnet", "aimnet2"):
            default_model = calculator_type
        model_name = _value_or_env(calculator_model, "DEEPCONF_AIMNET_MODEL", default_model)
        lig.setAIMNet2Calculator(
            model_name=model_name,
            charge=_parse_calculator_charge(
                calculator_charge,
                lig,
                calculator_label="AIMNet2",
                env_name="DEEPCONF_AIMNET_CHARGE",
            ),
            mult=calculator_mult,
        )
    elif "nequip" in calculator_type.lower():
        model_path = _resolve_nequip_model_path(calculator_model)
        lig.setNequIPCalculator(
            model_path=model_path,
            device=calculator_device,
            chemical_symbols=_resolve_nequip_chemical_symbols(
                nequip_chemical_symbols,
                model_path,
            ),
        )
    else:
        print(f"Unsupported calculator_type: {calculator_type}")
        sys.exit(1)

    if addH:
        if mmCalculator:
            lig.optimizeAddedHydrogensWithMM(maxiter=200)
        else:
            lig.optimizeAddedHydrogensWithCurrentCalculator(
                fmax=0.05,
                maxiter=200,
                opt_method="LBFGS",
            )

    # set optimizetion parameters
    lig.setOptParams(fmax=thr_fmax, maxiter=args.maxiter)

    if pre_optimization_lig:
        print("Pre-Optimization process.. before confromer generations")
        e = lig.geomOptimization()
        with open("%s/pre_%s%s_energy.txt"%(WORK_DIR, prefix, file_base) , "w") as pre_e_file:
            print(e, " eV", file=pre_e_file)
        lig.writeRWMol2File("%s/pre_%s%s.sdf"%(WORK_DIR, prefix, file_base), Energy=e)

    if genconformer:
        out_file_path="%s/%sminE_conformer.sdf"%(WORK_DIR, prefix)
        lig = setGenConformers(lig, out_file_path, mmCalculator)
        if lig is None:
            return None
        print("Conformer generation process is done")
    else:
        out_file_path="%s/global_%s%s.sdf"%(WORK_DIR, prefix, file_base)
        # geometry optimizaton for ligand
        if  optimization_lig:
            e = lig.geomOptimization()
            with open("%s/global_%s%s_energy.txt"%(WORK_DIR, prefix, file_base) , "w") as e_file:
                print(e, " eV", file=e_file)
            lig.writeRWMol2File("%s/global_%s%s.sdf"%(WORK_DIR, prefix, file_base), Energy=e)

    if not verbose:
        _export_final_sdf_and_cleanup(WORK_DIR, file_base, prefix)


if __name__ == "__main__":
    args = parser.parse_args()
    structure_dir = args.structure_dir
    calculator_type = args.calculator_type

    optimization_method = args.optimization_method

    optimization_conf = getBoolStr(args.optimization_conf)
    optimization_lig = getBoolStr(args.optimization_lig)
    pre_optimization_lig = getBoolStr(args.pre_optimization_lig)
    genconformer = getBoolStr(args.genconformer)
    ignore_hydrogen = getBoolStr(args.ignore_hydrogen)
    ETKDG = getBoolStr(args.ETKDG)

    nprocs = args.nprocs
    thr_fmax = args.thr_fmax
    maxiter = args.maxiter

    #get conformer generator parameters
    num_conformers = args.num_conformers
    max_attempts = args.max_attempts
    prune_rms_thresh = args.prune_rms_thresh
    opt_prune_rms_thresh = args.opt_prune_rms_thresh
    opt_prune_diffE_thresh = args.opt_prune_diffE_thresh
    nfold = args.nfold
    npick = args.npick
    nscale = args.nscale

    cluster_nprocs = args.cluster_nprocs
    cluster_chunk_size = args.cluster_chunk_size
    cluster_linkage = args.cluster_linkage.lower()
    organize_clusters = getBoolStr(args.organize_clusters)
    organize_mode = args.organize_mode.lower()
    summary_csv = args.summary_csv
    verbose = getBoolStr(args.verbose)
    calculator_model = args.calculator_model
    calculator_charge = args.calculator_charge
    calculator_mult = int(float(args.calculator_mult))
    calculator_device = args.calculator_device
    nequip_chemical_symbols = args.nequip_chemical_symbols
    g16_mem = args.g16_mem
    g16_level = args.g16_level
    g16_basis = args.g16_basis

    if not verbose:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.warning')

    file_names = [item for item in os.listdir(structure_dir) if not item.startswith(".")]
    failed_csv = open("failed_files.csv", "w")
    failed_csv.write("FileNames,\n")

    fl_timing = open("timings.csv", "w")
    print("FileName,Time(min.)", file=fl_timing)
    for file_name in file_names:
        file_base = file_name.split(".")[0]
        #  try:
        s_time = time.time()
        result = runConfGen(file_name)
        if result is None:
            continue

        print(file_name, ",", "%.4f"%((time.time()-s_time)/60), file=fl_timing)
        #  except:
        #      print("Error for %s file !!! Skipping...")
        #      failed_csv.write(file_name+",\n")
        #  break
    fl_timing.close()
    failed_csv.close()

