#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:38:14

@author: Javiera Jilberto Vallejos 
'''

import os
import src.FibGen as fg
import src.FibGenOld as fg_old
from time import time

run_flag = True
method = 'bayer'
svfsi_exec = "svmultiphysics "

mesh_path = "example/truncated/VOLUME.vtu"
surfaces_dir = f"example/truncated/mesh-surfaces"
outdir = "example/truncated/output"

surface_names = {'epi': 'EPI.vtp',
                 'epi_apex': 'EPI_APEX.vtp',    # New surface
                 'av': 'av.vtp',
                 'mv': 'mv.vtp',
                 'tv': 'tv.vtp',
                 'pv': 'pv.vtp',
                 'endo_lv': 'LV.vtp',
                 'endo_rv': 'RV.vtp'}

params = {
    # A = alpha angle
    'AENDORV' : 90,
    'AEPIRV' : -25,
    'AENDOLV' : 60,
    'AEPILV' : -60,

    # AOTENDOLV = 90
    'AOTENDOLV' : 0, # Changed 24/04/2020
    'AOTENDORV' : 90,
    'AOTEPILV' : 0,
    'AOTEPIRV' : 0,
    'ATRIENDO' : 0, # Changed 13/06/2020

    # B = beta angle
    'BENDORV' : 0,
    'BEPIRV' : 0,
    'BENDOLV' : 0,
    'BEPILV' : 0,
}

# Make sure the paths are full paths
mesh_path = os.path.abspath(mesh_path)
surfaces_dir = os.path.abspath(surfaces_dir)
outdir = os.path.abspath(outdir)

# Generate the apex surface
start = time()

start = time()
fg.generate_epi_apex(mesh_path, surfaces_dir, surface_names)

# # Run the Laplace solver
# run_flag = False
# outdir = os.path.join(os.path.dirname(outdir), "output")
# if run_flag:
#     if method == 'bayer':
#         template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "templates", "solver_bayer.xml")
#     laplace_results_file = fg.runLaplaceSolver(mesh_path, surfaces_dir, mesh_path, svfsi_exec, template_file, outdir, surface_names)
# laplace_results_file = outdir + '/result_001.vtu'

# # Generate the fiber directions
# result_mesh = fg.generate_fibers_BiV_Bayer_cells(outdir, laplace_results_file, params)

# print(f"generate fibers (new code) elapsed time: {time() - start:.3f} s")