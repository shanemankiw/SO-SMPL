import numpy as np
import trimesh

# Load the meshes
human_mesh = trimesh.load("smplx-male.obj")
vest_cut_mesh = trimesh.load("smplx-vestcut-6.obj")

# Store the vertex positions of both meshes in respective variables
human_vertices = human_mesh.vertices
vest_cut_vertices = vest_cut_mesh.vertices

# Assume that the positions haven't changed, only re-indexed
# Create a dictionary to map skirt vertex index to human vertex index
vertex_mapping = {}

# Loop through skirt vertices to find corresponding vertices in the human mesh
for i, remain_vertex in enumerate(vest_cut_vertices):
    # Find the index of this skirt vertex in the human vertices
    # We are assuming the position is the same, so we can use numpy's where function
    index = (human_vertices == remain_vertex).all(axis=1).nonzero()[0]
    if len(index) > 0:
        # Note that there can be multiple vertices with the same position due to split vertices on edge
        # We are just using the first one found here
        vertex_mapping[i] = index[0]

# Human mesh mask we want to transfer (For example purposes, assuming a simple mask here)
# In reality, this should be loaded or computed according to your specific scenario
human_mask = np.ones(len(human_mesh.vertices), dtype=int)
for human_index in vertex_mapping.values():
    human_mask[human_index] = 0

human_mask_np = np.asarray(human_mask)
np.save("vest_cut_mask_6.npy", human_mask_np)
