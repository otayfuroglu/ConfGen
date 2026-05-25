#! /usr/bin/env bash

ligPrep_DIR="$HOME/DeepConf"
PYTHON_DIR="$HOME/.local/Miniconda3/envs/ANI_AIMNet_NeQuIP/bin"

# Set this to your ligand/input directory.
struct_dir="./test"

# adding hydrogen if missing (yes/no) if yes, constrain heavy atoms and minimize hydrogens.
add_hydrogen=no

# set optimization method available in ASE (BFGS, LBFGS, GPMin, FIRE, Berny)
optimization_method=FIRE

# optimize ligand before conformer generation (yes/no)
pre_optimization_lig=no

# generate conformers (yes/no)
genconformer=yes

# conformer generator parameters. If ETKDG=yes, max_attempts and prune_rms_thresh are not used.
ETKDG=yes
num_conformers=50
max_attempts=100000

# RMSD threshold for initial RDKit conformer pruning.
prune_rms_thresh=0.05

# RMSD threshold for post-optimization conformer clustering.
opt_prune_rms_thresh=0.5

# Energy threshold for post-RMSD representative pruning.
opt_prune_diffE_thresh=0.01

# calculator type (ani1x/ani1ccx/ani2x/aimnet2/nequip/g16/uff)
caculator_type="aimnet2"

# Optional calculator settings:
# - AIMNet2: set calculator_model to aimnet2, aimnet2-2025, etc.
#   Leave calculator_charge blank to infer the SDF formal charge, or set an override.
# - Gaussian16: calculator_charge and calculator_multiplicity are passed to ASE Gaussian.
#   Leave calculator_charge blank to infer the SDF formal charge.
# - NequIP: set calculator_model to the deployed .pth path for older NequIP,
#   or to the compiled model path for newer NequIP.
#   If caculator_type="nequip" and calculator_model is blank, DeepConf uses
#   the bundled all_NNP_MODELS/G_NequIP.pth model with identity species mapping.
#   To use another bundled model, set calculator_model="$model_dir/$nequip_model_file".
model_dir="$ligPrep_DIR/all_NNP_MODELS"
nequip_model_file="G_NequIP.pth"
# Other available local examples:
# nequip_model_file="G_NequIP_smdW.pth"
# nequip_model_file="M_NequIP.pth"
# nequip_model_file="M_NequIP_smdO.pth"
# nequip_model_file="M_NequIP_smdW.pth"
calculator_model=""
calculator_charge=""
calculator_multiplicity=1
calculator_device=auto
nequip_chemical_symbols=""
g16_mem="4GB"
g16_level="WB97XD"
g16_basis="6-311++G(3df,3pd)"

# optimize generated conformers (yes/no)
optimization_conf=yes

# optimize original ligand only (yes/no)
optimization_lig=no

# number of processors for Gaussian and RMSD clustering defaults
nprocs=64

# optimization force threshold
thr_fmax=0.2

# maximum optimization iterations
maxiter=50000

# conformer sampling controls
nfold=2
npick=0
nscale=10

# optimized-conformer RMSD clustering controls
cluster_nprocs=64
cluster_chunk_size=4000
cluster_linkage=complete
organize_clusters=yes
organize_mode=move
summary_csv=cluster_summary.csv

# verbose=yes keeps all folders/intermediates.
# verbose=no exports <file_base>_output.sdf to this directory and removes the ligand work folder.
verbose=yes

"$PYTHON_DIR/python" "$ligPrep_DIR/runConfGen.py" \
"$struct_dir" \
"$add_hydrogen" \
"$caculator_type" \
"$optimization_method" \
"$optimization_conf" \
"$optimization_lig" \
"$pre_optimization_lig" \
"$genconformer" \
"$nprocs" \
"$thr_fmax" \
"$maxiter" \
"$ETKDG" \
"$num_conformers" \
"$max_attempts" \
"$prune_rms_thresh" \
"$opt_prune_rms_thresh" \
"$opt_prune_diffE_thresh" \
"$nfold" \
"$npick" \
"$nscale" \
"$cluster_nprocs" \
"$cluster_chunk_size" \
"$cluster_linkage" \
"$organize_clusters" \
"$organize_mode" \
"$summary_csv" \
"$verbose" \
"$calculator_model" \
"$calculator_charge" \
"$calculator_multiplicity" \
"$calculator_device" \
"$nequip_chemical_symbols" \
"$g16_mem" \
"$g16_level" \
"$g16_basis"
