#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/11/21 20:43:23

@author: Javiera Jilberto Vallejos 
'''

import os
import re
import numpy as np
import pyvista as pv



def get_normal_plane_svd(points):   # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid


def generate_epi_apex_jjv(mesh_path, surfaces_dir, surface_names):
    '''
    Generate the epi apex and epi mid surfaces from the epi surface of the BiV.
    
    Parameters:
    -----------
    surfaces_dir : str
        Directory containing surface meshes
    surface_names : list of str
        List of surface mesh filenames
    '''
    
    # Load the epi surface
    epi_name = os.path.join(surfaces_dir, surface_names['epi'])
    epi_mesh = pv.read(epi_name)
    epi_points = epi_mesh.points
    epi_cells = epi_mesh.faces
    epi_eNoN = epi_cells[0]
    epi_cells = epi_cells.reshape((-1, epi_eNoN + 1))
    epi_cells = epi_cells[:, 1:]
    epi_global_node_id = epi_mesh.point_data['GlobalNodeID']
    epi_global_cell_id = epi_mesh.cell_data['GlobalElementID']

    # Load the base surface
    base_name = os.path.join(surfaces_dir, surface_names['base'])
    base_mesh = pv.read(base_name)
    base_global_node_id = base_mesh.point_data['GlobalNodeID']

    # Extract the boundary of the epi surface (at the top) to find the apex point
    epi_base_global_node_id = np.intersect1d(epi_global_node_id, base_global_node_id)
    epi_base_nodes = np.where(np.isin(epi_global_node_id, epi_base_global_node_id))[0]

    # # Get normal
    base_normal, base_centroid = get_normal_plane_svd(epi_points[epi_base_nodes, :])

    # Find the index of the apex point of the epi surface
    distance = np.abs(base_normal@(epi_points - base_centroid).T)
    epi_apex_point_index = np.argmax(distance)

    # Find elements containing the apex point
    epi_apex_cell_index = np.where(epi_cells == epi_apex_point_index)[0]

    # Create epi_apex mesh
    submesh_cells = epi_cells[epi_apex_cell_index]
    submesh_xyz = np.zeros([len(np.unique(submesh_cells)), epi_points.shape[1]])
    map_mesh_submesh = np.ones(epi_points.shape[0], dtype=int)*-1
    map_submesh_mesh = np.zeros(submesh_xyz.shape[0], dtype=int)
    child_elems_new = np.zeros(submesh_cells.shape, dtype=int)

    cont = 0
    for e in range(submesh_cells.shape[0]):
        for i in range(submesh_cells.shape[1]):
            if map_mesh_submesh[submesh_cells[e,i]] == -1:
                child_elems_new[e,i] = cont
                submesh_xyz[cont] = epi_points[submesh_cells[e,i]]
                map_mesh_submesh[submesh_cells[e,i]] = cont
                map_submesh_mesh[cont] = submesh_cells[e,i]
                cont += 1
            else:
                child_elems_new[e,i] = map_mesh_submesh[submesh_cells[e,i]]

    epi_apex_cells_type = np.full((child_elems_new.shape[0], 1), epi_eNoN)
    epi_apex_cells = np.hstack((epi_apex_cells_type, child_elems_new))
    epi_apex_cells = np.hstack(epi_apex_cells) 

    # Get global IDs
    epi_apex_global_node_id = epi_global_node_id[map_submesh_mesh]
    epi_apex_global_cell_id = epi_global_cell_id[epi_apex_cell_index]

    # Create and save mesh
    epi_apex_mesh = pv.PolyData(submesh_xyz, epi_apex_cells)
    epi_apex_mesh.point_data.set_array(epi_apex_global_node_id, 'GlobalNodeID')
    epi_apex_mesh.cell_data.set_array(epi_apex_global_cell_id, 'GlobalElementID')

    epi_apex_name = os.path.join(surfaces_dir, surface_names['epi_apex'])
    epi_apex_mesh.save(epi_apex_name)


#################### OLD FUNCTION BELOW ####################

def distance_to_surface(point, surface_normal, surface_point):
    '''
    Calculate the distance from a point to a point on a surface
    '''
    # Ensure that the input vectors are NumPy arrays
    point = np.array(point)
    surface_normal = np.array(surface_normal)
    surface_point = np.array(surface_point)

    # Calculate the vector from the surface point to the target point
    v = point - surface_point

    # Calculate the dot product between the vector v and the surface normal
    dot_product = np.dot(v, surface_normal)

    # Calculate the absolute value of the dot product to get the distance
    distance = abs(dot_product)

    return distance

def generate_epi_apex(surfaces_dir, surface_names):
    '''
    Generate the epi apex and epi mid surfaces from the epi surface of the BiV.
    
    Parameters:
    -----------
    surfaces_dir : str
        Directory containing surface meshes
    surface_names : list of str
        List of surface mesh filenames
    '''

    # Generate epi_apex and epi_mid from epi surface
    # top_name = os.path.join(surfaces_dir, "top.vtp")
    # top_mesh = pv.read(top_name)
    # top_mesh = top_mesh.compute_normals()
    # #top_normal_rep = top_mesh['Normals'][0, :]
    # top_normal_rep = np.mean(top_mesh['Normals'], axis=0)
    # top_normal_rep = top_normal_rep / np.linalg.norm(top_normal_rep)
    # #top_point_rep = top_mesh.points[0, :]
    # top_point_rep = np.mean(top_mesh.points, axis=0)
    
    # Load the epi surface
    epi_name = os.path.join(surfaces_dir, surface_names['epi'])
    epi_mesh = pv.read(epi_name)
    epi_points = epi_mesh.points
    epi_cells = epi_mesh.faces
    epi_global_node_id = epi_mesh.point_data['GlobalNodeID']
    epi_global_cell_id = epi_mesh.cell_data['GlobalElementID']
    epi_eNoN = epi_cells[0]
    epi_cells = epi_cells.reshape((-1, epi_eNoN + 1))
    epi_cells = epi_cells[:, 1:]

    # Extract the boundary of the epi surface (at the top) to find the apex point
    epi_boundary = epi_mesh.extract_feature_edges(non_manifold_edges=False, feature_edges=False, manifold_edges=False)

    # Triangulate the boundary
    epi_boundary_triangulated = epi_boundary.delaunay_2d()

    # Compute the center and average normal vector of the triangulated boundary
    top_point_rep = np.mean(epi_boundary_triangulated.points, axis=0)
    epi_boundary_triangulated = epi_boundary_triangulated.compute_normals()
    top_normal_rep = np.mean(epi_boundary_triangulated['Normals'], axis=0)
    top_normal_rep = top_normal_rep / np.linalg.norm(top_normal_rep)
    print(top_normal_rep, top_point_rep)

    # # Show the normal vector and centroid
    # p = pv.Plotter()
    # p.add_mesh(epi_boundary_triangulated, color='white', show_edges=True)
    # p.add_arrows(top_point_rep, top_normal_rep, mag=20, color='r')
    # p.show()

    # Find the index of the apex point of the epi surface
    epi_apex_point_index = 0
    for i in range(epi_points.shape[0]):
        epi_point_dis_to_top = distance_to_surface(epi_points[i, :], top_normal_rep, top_point_rep)
        if i == 0:
            epi_apex_point_dis = epi_point_dis_to_top
        elif epi_point_dis_to_top > epi_apex_point_dis:
            epi_apex_point_index = i
            epi_apex_point_dis = epi_point_dis_to_top
    print("Epi apex point index:", epi_apex_point_index)

    # # Generate epi_apex mesh
    epi_apex_cell_index = np.where(epi_cells == epi_apex_point_index)[0]
    epi_apex_name = os.path.join(surfaces_dir, "epi_apex.vtp")
    epi_apex_cells_surf = epi_cells[epi_apex_cell_index, :]
    epi_apex_cells = np.zeros_like(epi_apex_cells_surf)
    epi_apex_points_index = np.unique(np.hstack(epi_apex_cells_surf))
    epi_apex_points = epi_points[epi_apex_points_index, :]
    for j in range(epi_apex_cells_surf.shape[0]):
        for k in range(epi_apex_cells_surf.shape[1]):
            temp_index = np.where(epi_apex_points_index == epi_apex_cells_surf[j, k])
            temp_index = temp_index[0]
            epi_apex_cells[j, k] = temp_index

    epi_apex_cells_type = np.full((epi_apex_cells.shape[0], 1), epi_eNoN)
    epi_apex_cells = np.hstack((epi_apex_cells_type, epi_apex_cells))
    epi_apex_cells = np.hstack(epi_apex_cells)

    epi_apex_global_node_id = epi_global_node_id[epi_apex_points_index]
    epi_apex_global_cell_id = epi_global_cell_id[epi_apex_cell_index]
    print(epi_apex_global_node_id)

    epi_apex_mesh = pv.PolyData(epi_apex_points, epi_apex_cells)

    epi_apex_mesh.point_data.set_array(epi_apex_global_node_id, 'GlobalNodeID')
    epi_apex_mesh.cell_data.set_array(epi_apex_global_cell_id, 'GlobalElementID')

    epi_apex_mesh.save(epi_apex_name)

    # # Generate epi_mid mesh (epi_mid = epi - epi_apex)
    # epi_mid_cell_index = np.where(~np.any(epi_cells == epi_apex_point_index, axis=1))[0]
    # epi_mid_name = os.path.join(surfaces_dir, "epi_mid.vtp")
    # epi_mid_cells_surf = epi_cells[epi_mid_cell_index, :]
    # epi_mid_cells = np.zeros_like(epi_mid_cells_surf)
    # epi_mid_points_index = np.unique(np.hstack(epi_mid_cells_surf))
    # epi_mid_points = epi_points[epi_mid_points_index, :]
    # for j in range(epi_mid_cells_surf.shape[0]):
    #     for k in range(epi_mid_cells_surf.shape[1]):
    #         temp_index = np.where(epi_mid_points_index == epi_mid_cells_surf[j, k])
    #         temp_index = temp_index[0]
    #         epi_mid_cells[j, k] = temp_index

    # epi_mid_cells_type = np.full((epi_mid_cells.shape[0], 1), epi_eNoN)
    # epi_mid_cells = np.hstack((epi_mid_cells_type, epi_mid_cells))
    # epi_mid_cells = np.hstack(epi_mid_cells)

    # epi_mid_global_node_id = epi_global_node_id[epi_mid_points_index]
    # epi_mid_global_cell_id = epi_global_cell_id[epi_mid_cell_index]

    # epi_mid_mesh = pv.PolyData(epi_mid_points, epi_mid_cells)

    # epi_mid_mesh.point_data.set_array(epi_mid_global_node_id, 'GlobalNodeID')
    # epi_mid_mesh.cell_data.set_array(epi_mid_global_cell_id, 'GlobalElementID')

    # epi_mid_mesh.save(epi_mid_name)


def runLaplaceSolver(mesh_dir, surfaces_dir, mesh_file, exec_svfsi, template_file, outdir, surface_names):
    xml_template_path = template_file
    out_name = os.path.join(surfaces_dir, "../svFSI_BiV.xml")
    
    with open(xml_template_path, 'r') as svFile:
        xml_content = svFile.read()
    
    # Update mesh file path using regex
    mesh_pattern = r'(<Mesh_file_path>)\s+[^\s<]+[^<]*(</Mesh_file_path>)'
    xml_content = re.sub(mesh_pattern, r'\1 ' + mesh_file + r' \2', xml_content)
    
    # Update face file paths - need to identify which face by checking context
    # Read lines to determine context
    lines = xml_content.split('\n')
    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has a face name
        face_match = re.search(r'name="([^"]+)"', line)
        face_name = face_match.group(1) if face_match else None
        
        # Look ahead for Face_file_path
        if face_name and i + 1 < len(lines) and "<Face_file_path>" in lines[i + 1]:
            # Determine which file to use based on face name
            if face_name == "epi":
                new_path = os.path.join(surfaces_dir, surface_names['epi'])
            elif face_name == "epi_top":
                new_path = os.path.join(surfaces_dir, surface_names['base'])
            elif face_name == "epi_apex":
                new_path = os.path.join(surfaces_dir, surface_names['epi_apex'])
            elif face_name == "endo_lv":
                new_path = os.path.join(surfaces_dir, surface_names['endo_lv'])
            elif face_name == "endo_rv":
                new_path = os.path.join(surfaces_dir, surface_names['endo_rv'])
            else:
                new_path = None
            
            if new_path:
                # Add current line
                updated_lines.append(line)
                # Replace the path in the next line
                i += 1
                face_pattern = r'(<Face_file_path>)\s+[^\s<]+[^<]*(</Face_file_path>)'
                updated_line = re.sub(face_pattern, r'\1 ' + new_path + r' \2', lines[i])
                updated_lines.append(updated_line)
                i += 1
                continue
        
        # Add line as-is
        updated_lines.append(line)
        i += 1
    
    xml_content = '\n'.join(updated_lines)
    
    # Update save results folder using regex
    save_pattern = r'(<Save_results_in_folder>)\s+[^\s<]+[^<]*(</Save_results_in_folder>)'
    xml_content = re.sub(save_pattern, r'\1 ' + outdir + r' \2', xml_content)

    with open(out_name, 'w') as svFileNew:
        svFileNew.write(xml_content)

    print("   Running svFSI solver")
    print(f"   {exec_svfsi + out_name}")
    os.system(exec_svfsi + out_name)

    return