#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:38:14

@author: Javiera Jilberto Vallejos 
'''

import os
import argparse
import src.FibGen as fg
from time import time

###########################################################
############  USER INPUTS  ################################
###########################################################

run_flag = True
svfsi_exec = "svmultiphysics "

mesh_path = "example/truncated/VOLUME.vtu"
surfaces_dir = None  # default computed relative to mesh_path below
outdir = "example/truncated/output_b"

surface_names = {'epi': 'EPI.vtp',
                 'epi_apex': 'EPI_APEX.vtp',    # New surface
                 'base': 'BASE.vtp',
                 'endo_lv': 'LV.vtp',
                 'endo_rv': 'RV.vtp'}

# Parameters for the Bayer et al. method https://doi.org/10.1007/s10439-012-0593-5. 
params = {
    "ALFA_END": 60.0,
    "ALFA_EPI": -60.0,
    "BETA_END": 20.0,
    "BETA_EPI": -20.0,
}


###########################################################
############  FIBER GENERATION  ###########################
###########################################################

# Optional CLI overrides
parser = argparse.ArgumentParser(description="Generate fibers using the Bayer method.")
parser.add_argument("--svfsi-exec", default=svfsi_exec, help="svMultiPhysics executable/command (default: %(default)s)")
parser.add_argument("--mesh-path", default=mesh_path, help="Path to the volumetric mesh .vtu (default: %(default)s)")
parser.add_argument(
    "--surfaces-dir",
    default=surfaces_dir,
    help="Directory containing mesh surfaces; default: <parent of mesh_path>/mesh-surfaces",
)
parser.add_argument("--outdir", default=outdir, help="Output directory (default: %(default)s)")
args = parser.parse_args()

svfsi_exec = args.svfsi_exec
if not svfsi_exec.endswith(" "):
    svfsi_exec = svfsi_exec + " "

mesh_path = args.mesh_path
outdir = args.outdir

# Make sure the paths are full paths
mesh_path = os.path.abspath(mesh_path)
outdir = os.path.abspath(outdir)

if args.surfaces_dir is None:
    surfaces_dir = os.path.join(os.path.dirname(mesh_path), "mesh-surfaces")
else:
    surfaces_dir = os.path.abspath(args.surfaces_dir)

start = time()
fg.generate_epi_apex(mesh_path, surfaces_dir, surface_names)

# Run the Laplace solver
if run_flag:
    template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer.xml")
    laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
laplace_results_file = outdir + '/result_001.vtu'

# Generate the fiber directions
result_mesh = fg.generate_fibers_BiV_Bayer_cells(outdir, laplace_results_file, params, return_angles=True, return_intermediate=True)

print(f"generate fibers (Bayer method) elapsed time: {time() - start:.3f} s")

# Optional, save the result mesh with intermediate field and angles for checking
result_mesh_path = os.path.join(outdir, "results_bayer.vtu")
result_mesh.save(result_mesh_path)
