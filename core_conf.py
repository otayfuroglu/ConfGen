#
from openbabel import openbabel, pybel
from ase import Atoms
from ase.io import write

import rdkit
from  rdkit import Chem
from  rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
#  import os_util
from collections import defaultdict
import sys, os, shutil
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from itertools import product, repeat
from functools import wraps

from scipy.cluster.vq import kmeans, vq, whiten
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from random import sample


NPROCS_ALL = int(cpu_count())
print("Number of total cpu core: ", NPROCS_ALL)


NEQUIP_ATOMIC_SELF_ENERGIES_EV = {
    "C": -1029.68639155054,
    "H": -13.6819685328354,
    "O": -2042.60847516334,
    "N": -1485.28074187715,
    "F": -2713.859759887,
    "S": -10833.1069601335,
    "P": -9286.00643051839,
    "Cl": -12521.2432312104,
}


def calcFuncRunTime(func):
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        s_time = time.time()
        func(*args, **kwargs)
        print(f"Function {func.__name__} executed in {(time.time()-s_time)/60:.5f} m")
    return wrapper


def calcRMSDsymm(pair_idx, mol_list):

    idx1 = pair_idx[0]
    idx2 = pair_idx[1]
    if idx1 < idx2:
        if len(mol_list) == 1:
            mol = mol_list[0]
            return (AllChem
                    .GetConformerRMS(mol,
                                     idx1,
                                     idx2,
                                     prealigned=False)
                   )
        else:
            mol1 = mol_list[idx1]
            mol2 = mol_list[idx2]
            try:
                return Chem.rdMolAlign.GetBestRMS(mol1, mol2)
            except RuntimeError:
                if mol1.GetNumAtoms() != mol2.GetNumAtoms():
                    name1 = mol1.GetProp("_Name") if mol1.HasProp("_Name") else str(idx1)
                    name2 = mol2.GetProp("_Name") if mol2.HasProp("_Name") else str(idx2)
                    raise RuntimeError(
                        f"RMSD failed and atom counts differ: {name1} "
                        f"({mol1.GetNumAtoms()}) vs {name2} ({mol2.GetNumAtoms()})"
                    )

                atom_map = [(i, i) for i in range(mol1.GetNumAtoms())]
                return Chem.rdMolAlign.AlignMol(mol2, mol1, atomMap=atom_map)


#  @calcFuncRunTime
def getDistMatrix(mol_list, conformerIds=None, nprocs=None, chunk_size=4000):

    n_mol=len(mol_list)
    if n_mol == 1 and conformerIds:
        n_mol = len(conformerIds)

    if n_mol <= 1:
        print("Clustering do not applied.. There is just one conformer")
        return None

    if nprocs is None or nprocs <= 0:
        nprocs = NPROCS_ALL

    print(f"RMSD matrix calculation using {nprocs} processes; pool chunksize={chunk_size}")

    with Pool(nprocs) as pool:
        results = pool.starmap(calcRMSDsymm,
                               zip(product(range(n_mol), repeat=2),
                                   repeat(mol_list)),
                               chunksize=chunk_size)

    ordered_all_rmsd = [result for result in results if result is not None]
    expected = n_mol * (n_mol - 1) // 2
    if len(ordered_all_rmsd) != expected:
        raise RuntimeError(
            f"RMSD matrix construction failed: expected {expected} pairwise "
            f"RMSD values, got {len(ordered_all_rmsd)}."
        )
    return symmetricize(n_mol, ordered_all_rmsd)


def symmetricize(n: int, list1D: list) -> np.array:

    dist_matrix=np.zeros(shape=(n, n))
    i = 0
    for idx1 in range(n):
        for idx2 in range(n):
            if idx1 == idx2:
                dist_matrix[idx1, idx2] = 0.0
            elif idx1 < idx2:
                dist_matrix[idx1, idx2] = list1D[i]
                i += 1
    return dist_matrix + dist_matrix.T



    def run(self, fmax=0.05, steps=100):
        """Run the optimization until convergence or maximum steps."""
        for step in range(self.maxiter):
            positions, energy = self.step(self.atoms.get_forces())
            max_force = np.linalg.norm(self.atoms.get_forces(), axis=1).max()

            if max_force < fmax:
                print(f"Converged at step {step + 1} with max force {max_force:.6f} eV/A.")
                break
            else:
                print(f"Step {step + 1}: Energy = {energy:.6f} eV, Max force = {max_force:.6f} eV/A.")
        else:
            print("Reached maximum number of iterations without convergence.")


class confGen:
    """

    """

    def __init__(self, mol_path, addH, WORK_DIR, verbose=True):
        self.mol_path = mol_path
        self.WORK_DIR = WORK_DIR
        self.verbose = bool(verbose)

        # for activete g16 optmization algorithm
        self.optG16 = False

        # set add missing H
        self.addH = addH

        # initialize calcultor
        self.calculator = None
        self.nequip_cohesive_energy = False

        # initialize RW mol
        self.rw_mol = None
        self._loadRWMol()

        # initialize geom opt paprameters
        self.maxiter = None
        self.fmax = None

        # initialize optimization method
        self.opt_method = None

        # trial number
        self.n_trial = 1

    def setVerbose(self, verbose=True):
        self.verbose = bool(verbose)
        return self

    def _optimizer_logfile(self):
        return '-' if getattr(self, "verbose", True) else None

    def optimizeAddedHydrogensWithCurrentCalculator(self, fmax=0.05, maxiter=200, opt_method="LBFGS"):
        if not self.addH:
            return self

        if self.calculator is None:
            print(
                "Warning: add_hydrogen=yes but no calculator is assigned yet; "
                "skipping fixed-heavy-atom H relaxation."
            )
            return self

        old_opt_method = self.opt_method
        old_fmax = self.fmax
        old_maxiter = self.maxiter

        if getattr(self, "verbose", True):
            print("Relaxing added hydrogens with fixed heavy atoms using the user-selected calculator.")

        try:
            self.setOptMethod(opt_method)
            self.setOptParams(fmax=fmax, maxiter=maxiter)
            self.geomOptimization(fix_heavy_atoms=True)
        finally:
            self.opt_method = old_opt_method
            self.fmax = old_fmax
            self.maxiter = old_maxiter

        return self

    def optimizeAddedHydrogensWithMM(self, maxiter=200):
        if not self.addH:
            return self

        if getattr(self, "verbose", True):
            print("Relaxing added hydrogens with fixed heavy atoms using RDKit MM.")

        mol = Chem.RWMol(self.rw_mol)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol)

        if ff is None:
            print(
                "Warning: RDKit MM force field could not be assigned; "
                "skipping fixed-heavy-atom H relaxation."
            )
            return self

        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":
                ff.AddFixedPoint(atom.GetIdx())

        ff.Minimize(maxIts=maxiter)
        self.rw_mol = mol
        return self

    def getFileBase(self):
        return self.mol_path.split("/")[-1].split(".")[0]

    def increaseTrialNum(self):
        self.n_trial += 1

    def _getFileFormat(self, file_path=None):
        if file_path:
            return file_path.split(".")[-1]

        return self.mol_path.split(".")[-1]

    def _loadRWMol(self):
        self._loadMolWithOB()

    def _loadMolWithRW(self, mol_path, sanitize=True):
        rd_mol = next(Chem.SDMolSupplier(mol_path, sanitize=sanitize, removeHs=False))
        if sanitize is False:
            rd_mol.UpdatePropertyCache(strict=False)
        self.rw_mol = Chem.RWMol(rd_mol)

    def _rdKekuleizeError(self, rd_mol):
        # correction  kekuleize error (especially N in aromatic ring)
        print("\nWarning!: There is kekulize error, ingnored sanitize and kekulized for N atom which is in aromatic ring\n")
        for i, atom in enumerate(rd_mol.GetAtoms()):
            if atom.GetSymbol() == "N" and atom.GetIsAromatic():
                print("Aromatic N atom idex: ",i+1)
                atom.SetNumExplicitHs(1)
        return rd_mol

    def _loadMolWithOB(self):

        pb_mol = next(pybel.readfile(self._getFileFormat(), self.mol_path))
        tmp_file_name = f"{self.WORK_DIR}/tmp_ob_file.sdf"

        #add hydrogen with openbabel
        if self.addH:
            if self._getFileFormat() == "xyz":
                print("Error: Cannot add hydrogen atoms to XYZ file format!!!")
                exit(1)

            if self._getFileFormat() != "sdf":
                # coorection sfg for add true Hydrogen
                #  print(self._getFileFormat())
                pb_mol.write("sdf", tmp_file_name, overwrite=True)

                corr_tmp_file_name = "corr_tmp_ob_file.sdf"
                corr_part = "  0  0  0  0  0  0  0  0  0  0"
                with open(corr_tmp_file_name, "w") as corr_sdf:
                    with open(tmp_file_name) as lines:
                        for line in lines:
                            if len(line) == 70:
                                line = line[:40] + corr_part + "\n"
                            corr_sdf.write(line)

                pb_mol = next(pybel.readfile("sdf", corr_tmp_file_name))
                self._rmFileExist(tmp_file_name)
                self._rmFileExist(corr_tmp_file_name)

            pb_mol.addh()
            pb_mol.make3D()

        #  # openbabel file to rdkit mol2 file
        pb_mol.write("sdf", tmp_file_name, overwrite=True)

        # laod as RW file
        try:
            self._loadMolWithRW(tmp_file_name)
        except:
            self._loadMolWithRW(tmp_file_name, sanitize=False)
            self.rw_mol = self._rdKekuleizeError(self.rw_mol)

        # Added hydrogens are relaxed after runConfGen.py assigns the
        # user-selected calculator. Do not hard-code ANI2x here.

    def addHwithRD(self):
        self.rw_mol = rdkit.Chem.rdmolops.AddHs(self.rw_mol, addCoords=True)

    def _rmFileExist(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def writeRWMol2File(self, file_path, **kwargs):

        # add missing H
        #  self.addHwithRD()

        file_format = self._getFileFormat(file_path)

        if file_format == "xyz":
            rdkit.Chem.rdmolfiles.MolToXYZFile(self.rw_mol, file_path)
        elif file_format == "pdb":
            rdkit.Chem.rdmolfiles.MolToPDBFile(self.rw_mol, file_path)
        elif file_format == "sdf":
            with Chem.rdmolfiles.SDWriter(file_path) as writer:
                for key, value in kwargs.items():
                    self.rw_mol.SetProp(key, str(value))
                    writer.write(self.rw_mol)

        else:
            print("Unknown file format")
            sys.exit(1)

    def _writeConf2File(self, mol, conformerId, file_path, **kwargs):
        with rdkit.Chem.SDWriter(file_path) as w:
            old_props = {
                key: mol.GetProp(key)
                for key in kwargs
                if mol.HasProp(key)
            }
            for key, value in kwargs.items():
                mol.SetProp(key, str(value))
            w.write(mol, conformerId)
            w.flush()
            w.close()
            for key in kwargs:
                if key in old_props:
                    mol.SetProp(key, old_props[key])
                else:
                    mol.ClearProp(key)

    def _getTorsionPoints(self):
        from rdkit.Chem import TorsionFingerprints
        torsion_points = []
        for torsions_list in TorsionFingerprints.CalculateTorsionLists(self.rw_mol):
            for torsions in torsions_list:
                if 180 in torsions:
                    torsion_points.append(torsions[0][0])
        return(torsion_points)

    def _getNumConfs(self, nfold, scaled=1):
        n_torsions = len(self._getTorsionPoints())
        if n_torsions >= 10:
            return 8096
        else:
            return int(scaled * nfold ** n_torsions)

    #  @calcFuncRunTime
    def _getClusterKmeansFromConfIds(self, conformerIds, dist_matrix, n_group):

        cluster_conf_id = defaultdict(list)
        whitened = whiten(dist_matrix)
        centroids, _ = kmeans(whitened, n_group)
        cluster, _ = vq(whitened,centroids)
        for key, value in zip(cluster, conformerIds):
            cluster_conf_id[key].append(value)

        return cluster_conf_id

    def _readSDFEnergy(self, file_path, default=np.inf):
        try:
            mol = next(Chem.SDMolSupplier(file_path, removeHs=False))
        except Exception:
            return default

        if mol is None or not mol.HasProp("Energy"):
            return default

        try:
            return float(mol.GetProp("Energy"))
        except Exception:
            return default

    #  @calcFuncRunTime
    def _getClusterRMSDFromFiles(self, conf_dir, rmsd_thresh, linkage_method="complete",
                                 cluster_nprocs=None, cluster_chunk_size=4000):
        sdf_files = sorted([fl_name for fl_name in os.listdir(conf_dir)
                            if fl_name.endswith(".sdf")])

        mol_list = []
        file_energy = {}
        for fl_name in sdf_files:
            sdf_path = f"{conf_dir}/{fl_name}"
            mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False))
            if mol is None:
                print(f"Warning: could not read {sdf_path}; skipping")
                continue

            mol.SetProp("_Name", fl_name)
            mol_list.append(mol)
            file_energy[fl_name] = self._readSDFEnergy(sdf_path)

        if len(mol_list) <= 1:
            print("Clustering do not applied.. There is just one conformer")
            return 0

        print(f"RMSD clustering optimized conformers with {linkage_method} linkage")
        print(f"RMSD threshold: {rmsd_thresh} Angstrom")

        dist_matrix = getDistMatrix(
            mol_list,
            conformerIds=None,
            nprocs=cluster_nprocs,
            chunk_size=cluster_chunk_size,
        )
        if dist_matrix is None:
            return 0

        condensed_dist_matrix = squareform(dist_matrix, checks=False)
        linked = linkage(condensed_dist_matrix, method=linkage_method)

        raw_cluster_conf = defaultdict(list)
        labelList = [mol.GetProp('_Name') for mol in mol_list]
        for key, fl_name in zip(fcluster(linked, rmsd_thresh, criterion='distance'), labelList):
            raw_cluster_conf[key].append(fl_name)

        cluster_records = []
        for raw_cluster_id, fl_names in raw_cluster_conf.items():
            fl_names_sorted = sorted(fl_names, key=lambda f: file_energy.get(f, np.inf))
            rep_file = fl_names_sorted[0]
            rep_energy = file_energy.get(rep_file, np.inf)
            cluster_records.append((raw_cluster_id, rep_file, rep_energy, fl_names_sorted))

        cluster_records.sort(key=lambda x: x[2])

        cluster_conf = defaultdict(list)
        for new_cluster_id, (_, rep_file, rep_energy, fl_names_sorted) in enumerate(cluster_records, start=1):
            cluster_conf[new_cluster_id] = fl_names_sorted
            print(f"cluster_{new_cluster_id}: representative={rep_file}, Energy={rep_energy}, size={len(fl_names_sorted)}")

        return cluster_conf

    def _getCluster_diffE(self, files_minE, diffE_thresh=0.001):
        n_files = len(files_minE)
        dist_matrix = np.zeros(shape=(n_files, n_files))
        for i , val1 in enumerate(files_minE.values()):
            for j, val2 in  enumerate(files_minE.values()):
                e_diff = val1 - val2
                dist_matrix[i, j] = abs(val1 - val2 )
        linked = linkage(squareform(dist_matrix, checks=False), 'complete')
        label_list = list(files_minE.keys())
        cluster_conf = defaultdict(list)
        for key, fl_name in zip(fcluster(linked, diffE_thresh, criterion='distance'), label_list):
            cluster_conf[key].append(fl_name)
        return cluster_conf

    def _findConformerFileAfterOrganization(self, conf_dir, fl_name, cluster_conf=None):
        direct_path = f"{conf_dir}/{fl_name}"
        if os.path.exists(direct_path):
            return direct_path

        if cluster_conf is not None:
            for cluster_id, fl_names in cluster_conf.items():
                if fl_name in fl_names:
                    cluster_path = f"{conf_dir}/cluster_{cluster_id}/{fl_name}"
                    if os.path.exists(cluster_path):
                        return cluster_path

        for root, dirs, files in os.walk(conf_dir):
            if fl_name in files:
                return os.path.join(root, fl_name)

        return None

    def _writeEnergyRankedRepresentativeSDF(self, conf_dir, selected_sorted, cluster_conf=None):
        output_sdf = f"{conf_dir}/{self.getFileBase()}_output.sdf"

        if os.path.exists(output_sdf):
            os.remove(output_sdf)

        nwritten = 0
        with Chem.SDWriter(output_sdf) as w:
            for rank, (fl_name, e) in enumerate(selected_sorted.items(), start=1):
                mol_path = self._findConformerFileAfterOrganization(
                    conf_dir,
                    fl_name,
                    cluster_conf=cluster_conf,
                )
                if mol_path is None:
                    print(f"Warning: representative source SDF not found for {fl_name}; skipping")
                    continue

                mol = next(Chem.SDMolSupplier(mol_path, removeHs=False))
                if mol is None:
                    print(f"Warning: could not read representative SDF: {mol_path}")
                    continue

                mol.SetProp("Energy", str(e))
                mol.SetProp("_Name", fl_name)
                mol.SetProp("RepresentativeRankByEnergy", str(rank))
                mol.SetProp("SourceFile", fl_name)
                w.write(mol)
                nwritten += 1

        if nwritten == 0:
            print(f"Warning: no molecules were written to representative SDF: {output_sdf}")
        else:
            print(f"Representative SDF written to: {output_sdf} ({nwritten} structures)")

        return output_sdf

    def _pruneOptConfs(self, cluster_conf, confs_energies, conf_dir, opt_prune_diffE_thresh,
                       organize_clusters=True, organize_mode="move", summary_csv="cluster_summary.csv"):
        print("Applied diff RMSD filter (Angstrom)")

        energy_by_file = {}
        for _, row in confs_energies.iterrows():
            energy_by_file[row["FileName"]] = float(row["Energy(eV)"])

        local_files_minE = {}
        global_minE = None
        global_minE_file = None

        for cluster_id, fl_names in cluster_conf.items():
            fl_names.sort(key=lambda f: energy_by_file[f])
            minE_file = fl_names[0]
            minE = energy_by_file[minE_file]
            local_files_minE[minE_file] = minE

            if global_minE is None or minE < global_minE:
                global_minE = minE
                global_minE_file = minE_file

            print(f"cluster_{cluster_id}: representative={minE_file}, Energy={minE}, size={len(fl_names)}")

        selected_after_energy_filter = dict(local_files_minE)

        if len(selected_after_energy_filter) > 1:
            print("Applied diff Energy filter (eV/Atom)")
            energy_cluster_conf = self._getCluster_diffE(
                selected_after_energy_filter,
                diffE_thresh=opt_prune_diffE_thresh,
            )
            for fl_names in energy_cluster_conf.values():
                if len(fl_names) > 1:
                    fl_names.sort(key=lambda f: selected_after_energy_filter[f])
                    keep_file = fl_names[0]
                    if global_minE_file in fl_names:
                        keep_file = global_minE_file

                    for fl_name in fl_names:
                        if fl_name == keep_file:
                            continue
                        print("Energy-pruned representative", fl_name)
                        if fl_name in selected_after_energy_filter:
                            del selected_after_energy_filter[fl_name]

        selected_sorted = dict(sorted(selected_after_energy_filter.items(), key=lambda item: item[1]))

        self._writeEnergyRankedRepresentativeSDF(
            conf_dir,
            selected_sorted,
            cluster_conf=cluster_conf,
        )

        summary_rows = []
        for cluster_id, fl_names in cluster_conf.items():
            cluster_dir_name = f"cluster_{cluster_id}"
            directory = f"{conf_dir}/{cluster_dir_name}"

            if organize_clusters:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.mkdir(directory)

            for fl_name in fl_names:
                src = f"{conf_dir}/{fl_name}"
                dst = f"{directory}/{fl_name}"
                is_rmsd_rep = fl_name in local_files_minE
                is_final_rep = fl_name in selected_sorted

                summary_rows.append({
                    "ClusterRank": cluster_id,
                    "ClusterDir": cluster_dir_name if organize_clusters else "",
                    "FileName": fl_name,
                    "Energy(eV)": energy_by_file[fl_name],
                    "RMSDRepresentative": is_rmsd_rep,
                    "FinalRepresentative": is_final_rep,
                })

                if organize_clusters:
                    if os.path.exists(src):
                        if organize_mode == "move":
                            shutil.move(src, dst)
                        elif organize_mode == "copy":
                            shutil.copy2(src, dst)
                        else:
                            raise ValueError("organize_mode must be 'move' or 'copy'")
                    else:
                        print(f"Warning: source file not found during cluster organization: {src}")

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(["ClusterRank", "Energy(eV)"])

        if summary_csv in (None, "", "none", "None", "NO", "no"):
            summary_csv_path = f"{conf_dir}/{self.getFileBase()}_cluster_summary.csv"
        elif os.path.isabs(summary_csv):
            summary_csv_path = summary_csv
        else:
            summary_csv_path = f"{conf_dir}/{summary_csv}"

        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Cluster summary written to: {summary_csv_path}")

        self._writeEnergyRankedRepresentativeSDF(
            conf_dir,
            selected_sorted,
            cluster_conf=cluster_conf,
        )

    def genGonformers(self, file_path,
                         numConfs=100,
                         ETKDG=False,
                         maxAttempts=10000,
                         pruneRmsThresh=0.1,
                         mmCalculator=False,
                         optimization_conf=False,
                         opt_prune_rms_thresh=0.2,
                         opt_prune_diffE_thresh=0.001,
                         saveConfs=True,
                         useExpTorsionAnglePrefs=True,
                         useBasicKnowledge=True,
                         enforceChirality=True,
                         nfold=2,
                         npick=2,
                         nscale=1,
                         cluster_nprocs=None,
                         cluster_chunk_size=4000,
                         cluster_linkage="complete",
                         organize_clusters=True,
                         organize_mode="move",
                         summary_csv="cluster_summary.csv",
                        ):

        import copy

        if not getattr(self, "verbose", True):
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.warning')

        #  self.addHwithRD()
        print("Working on conformer generation process")
        mol = copy.deepcopy(self.rw_mol)
        if numConfs == 0 or numConfs < self._getNumConfs(nfold, scaled=nscale):
            numConfs = self._getNumConfs(nfold, scaled=nscale)
            print(f"Maximum number of conformers setting to {numConfs}")

        if ETKDG:
            ps = rdkit.Chem.rdDistGeom.ETKDGv3()
            conformerIds = list(rdkit.Chem.rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs,
                ps
            ))
        else:
            conformerIds = list(rdkit.Chem.AllChem.EmbedMultipleConfs(
                mol,
                numConfs=numConfs,
                maxAttempts=maxAttempts,
                pruneRmsThresh=pruneRmsThresh,
                useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                useBasicKnowledge=useBasicKnowledge,
                enforceChirality=enforceChirality,
                numThreads=NPROCS_ALL,
            ))

        # file for saving energies
        file_csv = open("%s/all_confs_sp_energies.csv" %self.WORK_DIR, "w")
        print("FileName,Energy(eV)", file=file_csv)
        file_csv.flush()

        print("Number of generated conformation: %d" %len(conformerIds))

        #  for k-means clutering
        #  dist_matrix = self._getConfDistMatrix(mol, conformerIds)
        print("Obtaining pairwise distance distribution matrix")
        dist_matrix = getDistMatrix([mol], conformerIds)

        print("Processing k-means clustering")
        cluster_conf_id = self._getClusterKmeansFromConfIds(conformerIds, dist_matrix,
                                           n_group=self._getNumConfs(nfold, scaled=1)
                                          )
        print("Calculating SP energies")
        minEConformerIDs = []
        all_picked_confs = []

        for cluster, clustered_confIds in cluster_conf_id.items():

            if saveConfs:
                CONF_DIR = self.WORK_DIR + f"/confs_cluster_{cluster}"
                if not os.path.exists(CONF_DIR):
                    os.mkdir(CONF_DIR)

            for i, conformerId  in enumerate(clustered_confIds):
                #  print("%d. conformer processing..." %i)
                if saveConfs:
                    prefix = ""
                    conf_file_path = "%s/conf_%d.sdf"%(CONF_DIR, conformerId)

                #create ase atoms
                ase_atoms = self._rwConformer2AseAtoms(mol, conformerId)
                if mmCalculator:
                    e = self._calcEnergyWithMM(mol, conformerId, 100)["energy_abs"]
                else:
                    e, _ = self._calcSPEnergy(mol, conformerId)

                if saveConfs:
                    self._writeConf2File(mol, conformerId, conf_file_path, Energy=e)

                if i == 0:
                    minE = e
                    minEConformerID = conformerId
                    minE_ase_atoms = ase_atoms
                else:
                    if minE > e:
                        minE = e
                        minEConformerID = conformerId
                        minE_ase_atoms = ase_atoms
                print("%sconf_%d.sdf,%s"%(prefix, conformerId, e), file=file_csv)
                file_csv.flush()

            minEConformerIDs.append(minEConformerID)

            # to pick conformer randomly
            picked = False
            while picked is False:
                rndConformerIDs = sample(clustered_confIds, npick)
                if minEConformerID not in rndConformerIDs:
                    picked = True

            picked_confs = [minEConformerID] + rndConformerIDs
            all_picked_confs += picked_confs

        # test
        assert len(minEConformerIDs) == len(cluster_conf_id.keys())
        # close to csv file
        file_csv.close()

        prefix = ""
        PICKED_CONF_DIR = self.WORK_DIR
        if optimization_conf:
            prefix = "opt_"

        PICKED_CONF_DIR = self.WORK_DIR + f"/{prefix}picked_confs"
        if not os.path.exists(PICKED_CONF_DIR):
            os.mkdir(PICKED_CONF_DIR)
        picked_file_csv = open(f"{PICKED_CONF_DIR}/{prefix}picked_confs_energies.csv", "w")
        print("FileName,Energy(eV),EnergyPerAtom(eV)", file=picked_file_csv)
        picked_file_csv.flush()

        for i, conformerId  in enumerate(all_picked_confs):
            if optimization_conf:
                e, ase_atoms = self._geomOptimizationConf(mol, conformerId)
            else:
                e, ase_atoms = self._calcSPEnergy(mol, conformerId)
            conf_file_path = "%s/%sconf_%d.sdf"%(PICKED_CONF_DIR, prefix, conformerId)

            #  save optimized structure  with rdkit as sdf
            with Chem.rdmolfiles.SDWriter(conf_file_path) as writer:
                rwmol = self.aseAtoms2rwMol(ase_atoms, template_mol=mol)
                rwmol.SetProp("Energy", str(e))
                rwmol.SetProp("_Name", f"{prefix}conf_{conformerId}")
                writer.write(rwmol)

            print("%sconf_%d.sdf,%s,%s"%(prefix,
                                         conformerId,
                                         e,
                                         e/len(ase_atoms)),
                  file=picked_file_csv)
            picked_file_csv.flush()
        picked_file_csv.close()

        # cluster and prune opitimzed confs by RMSD
        if optimization_conf:
            confs_energies = pd.read_csv(f"{PICKED_CONF_DIR}/{prefix}picked_confs_energies.csv")
            #  print(confs_energies)
            cluster_conf = self._getClusterRMSDFromFiles(
                PICKED_CONF_DIR,
                rmsd_thresh=opt_prune_rms_thresh,
                linkage_method=cluster_linkage,
                cluster_nprocs=cluster_nprocs,
                cluster_chunk_size=cluster_chunk_size,
            )
            if cluster_conf != 0:
                self._pruneOptConfs(
                    cluster_conf,
                    confs_energies,
                    PICKED_CONF_DIR,
                    opt_prune_diffE_thresh,
                    organize_clusters=organize_clusters,
                    organize_mode=organize_mode,
                    summary_csv=summary_csv,
                )
            else:
                src = f"{PICKED_CONF_DIR}/{confs_energies['FileName'][0]}"
                dst = f"{PICKED_CONF_DIR}/{self.getFileBase()}_output.sdf"
                os.rename(src, dst)
        else:
            confs_energies = pd.read_csv(f"{PICKED_CONF_DIR}/{prefix}picked_confs_energies.csv")
            confs_energies = confs_energies.sort_values("Energy(eV)")
            output_sdf = f"{PICKED_CONF_DIR}/{self.getFileBase()}_output.sdf"
            with Chem.SDWriter(output_sdf) as w:
                for rank, (_, row) in enumerate(confs_energies.iterrows(), start=1):
                    fl_name = row["FileName"]
                    e = float(row["Energy(eV)"])
                    mol_path = f"{PICKED_CONF_DIR}/{fl_name}"
                    mol_out = next(Chem.SDMolSupplier(mol_path, removeHs=False))
                    if mol_out is None:
                        continue
                    mol_out.SetProp("Energy", str(e))
                    mol_out.SetProp("_Name", fl_name)
                    mol_out.SetProp("RepresentativeRankByEnergy", str(rank))
                    w.write(mol_out)
            print(f"Representative SDF written to: {output_sdf}")

    def _calcEnergyWithMM(self, mol, conformerId, minimizeIts):
        ff = rdkit.Chem.AllChem.MMFFGetMoleculeForceField(
            mol,
            rdkit.Chem.AllChem.MMFFGetMoleculeProperties(mol),
            confId=conformerId)
        ff.Initialize()
        ff.CalcEnergy()
        results = {}
        if minimizeIts > 0:
            results["converged"] = ff.Minimize(maxIts=minimizeIts)
        results["energy_abs"] = ff.CalcEnergy()
        return results

    def setG16Calculator(self, label, chk, nprocs, xc, basis, scf, addsec=None,
                         extra=None, charge=0, mult=1, mem="4GB"):
        from ase.calculators.gaussian import Gaussian
        self.optG16 = True
        self.nequip_cohesive_energy = False

        self.calculator = Gaussian(
            label=label,
            #  chk=chk,
            nprocshared=nprocs,
            xc=xc,
            basis=basis,
            scf=scf,
            addsec=addsec,
            extra=extra,
            charge=charge,
            mult=mult,
            mem=mem,
        )

    def setANI2XCalculator(self):
        self.setANICalculator("ani2x")

    def setANICalculator(self, model_name="ani2x"):
        import torchani
        import torch
        self.nequip_cohesive_energy = False
        if getattr(self, "verbose", True):
            print("Number of CUDA devices: ", torch.cuda.device_count())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_key = model_name.lower().replace("-", "").replace("_", "")
        model_factories = {
            "ani1x": torchani.models.ANI1x,
            "ani1ccx": torchani.models.ANI1ccx,
            "ani2x": torchani.models.ANI2x,
        }
        if model_key not in model_factories:
            raise ValueError("ANI model must be one of ani1x, ani1ccx, or ani2x")

        self.calculator = model_factories[model_key]().to(device).ase()

    def setAIMNet2Calculator(self, model_name="aimnet2", charge=0, mult=1):
        self.nequip_cohesive_energy = False
        try:
            from aimnet.calculators import AIMNet2ASE
        except ImportError:
            from aimnet.calculators.aimnet2ase import AIMNet2ASE

        self.calculator = AIMNet2ASE(model_name, charge=charge, mult=mult)

    def setNequIPCalculator(self, model_path, device="auto", chemical_symbols=None):
        import inspect
        import torch
        from nequip.ase import NequIPCalculator

        if device in ("", "auto", None):
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not model_path:
            raise ValueError("NequIP calculator requires a compiled/deployed model path")

        if getattr(self, "verbose", True):
            print(f"Using NequIP device: {device}")

        model_ext = str(model_path).lower()
        prefer_deployed = model_ext.endswith((".pth", ".pt"))

        if prefer_deployed and hasattr(NequIPCalculator, "from_deployed_model"):
            method = NequIPCalculator.from_deployed_model
        elif hasattr(NequIPCalculator, "from_compiled_model"):
            method = NequIPCalculator.from_compiled_model
        elif hasattr(NequIPCalculator, "from_deployed_model"):
            method = NequIPCalculator.from_deployed_model
        else:
            raise AttributeError("NequIPCalculator has no supported model-loading method")

        kwargs = {}
        params = inspect.signature(method).parameters
        if "compile_path" in params:
            kwargs["compile_path"] = model_path
        elif "model_path" in params:
            kwargs["model_path"] = model_path
        else:
            kwargs[next(iter(params))] = model_path

        if "device" in params:
            kwargs["device"] = device

        if chemical_symbols:
            if chemical_symbols is True and "chemical_species_to_atom_type_map" in params:
                kwargs["chemical_species_to_atom_type_map"] = True
            elif chemical_symbols is True:
                pass
            elif "chemical_symbols" in params:
                kwargs["chemical_symbols"] = chemical_symbols
            elif "chemical_species_to_atom_type_map" in params:
                kwargs["chemical_species_to_atom_type_map"] = chemical_symbols
            elif "species_to_type_name" in params:
                kwargs["species_to_type_name"] = chemical_symbols

        self.calculator = method(**kwargs)
        self.nequip_cohesive_energy = True
        if getattr(self, "verbose", True):
            print("NequIP energies will be written as total energies after adding atomic self energies.")

    def _nequip_atomic_self_energy(self, ase_atoms):
        total_self_energy = 0.0
        missing_symbols = []
        for atom in ase_atoms:
            try:
                total_self_energy += NEQUIP_ATOMIC_SELF_ENERGIES_EV[atom.symbol]
            except KeyError:
                missing_symbols.append(atom.symbol)

        if missing_symbols:
            missing = ", ".join(sorted(set(missing_symbols)))
            raise ValueError(
                "Missing NequIP atomic self-energy reference for element(s): "
                f"{missing}"
            )

        return total_self_energy

    def _reportedCalculatorEnergy(self, ase_atoms, calculator_energy):
        if getattr(self, "nequip_cohesive_energy", False):
            return float(calculator_energy) + self._nequip_atomic_self_energy(ase_atoms)

        return calculator_energy

    def _calcSPEnergy(self, mol, conformerId):

        if self.calculator is None:
            print("Error: Calculator not found. Please set any calculator")
            sys.exit(1)

        ase_atoms = self._rwConformer2AseAtoms(mol, conformerId)
        #  from ase.io import write
        #  write("test_ase_atoms.xyz", ase_atoms)
        ase_atoms.calc = self.calculator

        e = self._reportedCalculatorEnergy(ase_atoms, ase_atoms.get_potential_energy())
        return e, ase_atoms

    def calcSPEnergy(self):

        if self.calculator is None:
            print("Error: Calculator not found. Please set any calculator")
            sys.exit(1)
        ase_atoms= self.rwMol2AseAtoms()
        ase_atoms.calc = self.calculator
        return self._reportedCalculatorEnergy(ase_atoms, ase_atoms.get_potential_energy())

    def setOptParams(self, fmax, maxiter):
        self.maxiter = maxiter
        self.fmax = fmax

    def setOptMethod(self, opt_method):
        self.opt_method = opt_method.lower()

    def _getOptMethod(self, ase_atoms):
        if self.opt_method is None or self.opt_method=="lbfgs":
            from ase.optimize import LBFGS
            return LBFGS(ase_atoms, logfile=self._optimizer_logfile())
        elif self.opt_method=="bfgs":
            from ase.optimize import BFGS
            return BFGS(ase_atoms, logfile=self._optimizer_logfile())
        elif self.opt_method=="fire":
            from ase.optimize import FIRE
            return FIRE(ase_atoms, logfile=self._optimizer_logfile())
        elif self.opt_method=="gpmin":
            from ase.optimize import GPMin
            return GPMin(ase_atoms, logfile=self._optimizer_logfile())
        elif self.opt_method=="berny":
            from ase.optimize import Berny
            return Berny(ase_atoms)
        elif self.opt_method=="cg":
            from ase.optimize.sciopt import SciPyFminCG
            return SciPyFminCG(ase_atoms)
        elif self.opt_method=="newtonraphson":
            from ase_optmizer_newton_raphson import NewtonRaphson
            return NewtonRaphson(ase_atoms)

    def _geomOptimizationConf(self, mol, conformerId):
        from ase.calculators.gaussian import GaussianOptimizer, Gaussian

        if self.calculator is None:
            print("Error: Calculator not found. Please set any calculator")
            sys.exit(1)


        ase_atoms = self._rwConformer2AseAtoms(mol, conformerId)
        #  from ase.io import write
        #  write("test_ase_atoms.xyz", ase_atoms)

        if self.fmax is None or self.maxiter is None:
            print("Error setting geometry optimizatian parameters for ASE. Please do it")
            exit(1)

        if self.optG16:
            dyn =  GaussianOptimizer(ase_atoms, self.calculator)
            dyn.run(steps=self.maxiter)
        else:
            ase_atoms.calc = self.calculator
            dyn = self._getOptMethod(ase_atoms)
            dyn.run(fmax=self.fmax, steps=self.maxiter)

        e = self._reportedCalculatorEnergy(ase_atoms, ase_atoms.get_potential_energy())
        return e, ase_atoms

    def geomOptimization(self, fix_heavy_atoms=False):
        from ase.calculators.gaussian import GaussianOptimizer

        if self.calculator is None:
            print("Error: Calculator not found. Please set any calculator")
            sys.exit(1)
        if self.fmax is None or self.maxiter is None:
            print("Error setting geometry optimizatian parameters for ASE. Please do it")
            exit(1)

        ase_atoms = self.rwMol2AseAtoms()
        if fix_heavy_atoms:
            from ase.constraints import FixAtoms
            c = FixAtoms(indices=[atom.index for atom in ase_atoms if atom.symbol != 'H'])
            ase_atoms.set_constraint(c)

        if self.optG16:
            dyn =  GaussianOptimizer(ase_atoms, self.calculator)
            dyn.run(steps=self.maxiter)
        else:
            ase_atoms.calc = self.calculator
            #  self.dyn = LBFGS(ase_atoms)
            dyn = self._getOptMethod(ase_atoms)
            dyn.run(fmax=self.fmax, steps=self.maxiter)

        self.rw_mol = self.aseAtoms2rwMol(ase_atoms, template_mol=self.rw_mol)
        return self._reportedCalculatorEnergy(ase_atoms, ase_atoms.get_potential_energy())

    def _rwConformer2AseAtoms(self, mol, conformerId):

        mol = mol.GetConformer(conformerId)

        atom_species = [atom.GetAtomicNum() for atom in mol.GetOwningMol().GetAtoms()]
        positions = mol.GetPositions()

        return Atoms(atom_species, positions)

    def rwMol2AseAtoms(self):

        atom_species = [atom.GetAtomicNum() for atom in self.rw_mol.GetAtoms()]

        conf = self.rw_mol.GetConformer()
        positions = [conf.GetAtomPosition(i) for i in range(len(atom_species))]

        return Atoms(atom_species, positions)

    def obMol2AseAtoms(self, ob_mol):
        from ase import Atom
        ase_atoms = Atoms()
        for i in range(ob_mol.NumAtoms()):
            obatom = ob_mol.GetAtom(i + 1)
            ase_atoms.append(Atom(obatom.GetAtomicNum(),
                              [obatom.GetX(),
                               obatom.GetY(),
                               obatom.GetZ()]
                             ))
        return ase_atoms

    def aseAtoms2rwMol(self, ase_atoms, template_mol=None):
        """
        Preserve the RDKit topology and replace only coordinates from ASE.

        The fixed topology is used only so RMSD clustering compares conformers
        with a consistent atom/bond graph. Coordinates remain the optimized ASE
        coordinates, so proton-transfer-like geometries are still written.
        """

        if template_mol is None:
            template_mol = self.rw_mol

        if template_mol is None:
            raise ValueError("template_mol is None; cannot preserve RDKit topology.")

        n_atoms_rdkit = template_mol.GetNumAtoms()
        n_atoms_ase = len(ase_atoms)
        if n_atoms_rdkit != n_atoms_ase:
            raise ValueError(
                f"Atom count mismatch while converting ASE->RDKit: "
                f"template has {n_atoms_rdkit}, ASE has {n_atoms_ase}"
            )

        rd_mol = Chem.Mol(template_mol)
        rd_mol.RemoveAllConformers()

        conf = Chem.Conformer(n_atoms_rdkit)
        positions = ase_atoms.get_positions()
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(
                i,
                Point3D(float(pos[0]), float(pos[1]), float(pos[2]))
            )

        rd_mol.AddConformer(conf, assignId=True)
        return Chem.RWMol(rd_mol)


    def writeAseAtoms(self, file_path):
        ase_atoms = self.rwMol2AseAtoms()

        # write mol to xyz file by ase
        write(file_path, ase_atoms)


