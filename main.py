#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:38:14

@author: Javiera Jilberto Vallejos 
'''

import os
import src.FibGen as fg
from time import time

run_flag = True
method = 'bayer'
svfsi_exec = "svmultiphysics "

mesh_path = "example/truncated/VOLUME.vtu"
surfaces_dir = f"example/truncated/mesh-surfaces"
outdir = "example/truncated/output"

surface_names = {'epi': 'EPI.vtp',
                 'epi_apex': 'EPI_APEX2.vtp',    # New surface
                 'base': 'BASE.vtp',
                 'endo_lv': 'LV.vtp',
                 'endo_rv': 'RV.vtp'}

params = {
    "ALFA_END": 60.0,
    "ALFA_EPI": -60.0,
    "BETA_END": 20.0,
    "BETA_EPI": -20.0,
}

# Make sure the paths are full paths
mesh_path = os.path.abspath(mesh_path)
surfaces_dir = os.path.abspath(surfaces_dir)
outdir = os.path.abspath(outdir)

# Generate the apex surface
start = time()
fg.generate_epi_apex(surfaces_dir, surface_names)
print(f"generate_epi_apex elapsed: {time() - start:.3f} s")

start = time()
fg.generate_epi_apex_jjv(mesh_path, surfaces_dir, surface_names)
print(f"generate_epi_apex_jjv elapsed: {time() - start:.3f} s")


# Run the Laplace solver
if run_flag:
    if method == 'bayer':
        template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer.xml")
    fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)