import numpy as np
import trimesh

# Load the meshes
human_mesh = trimesh.load("load/smplx/retopology/human.obj")
skirts_mesh = trimesh.load("load/smplx/retopology/skirts.obj")

# Store the vertex positions of both meshes in respective variables
human_vertices = human_mesh.vertices
skirts_vertices = skirts_mesh.vertices

# Assume that the positions haven't changed, only re-indexed
# Create a dictionary to map skirt vertex index to human vertex index
vertex_mapping = {}

# Loop through skirt vertices to find corresponding vertices in the human mesh
for i, skirt_vertex in enumerate(skirts_vertices):
    # Find the index of this skirt vertex in the human vertices
    # We are assuming the position is the same, so we can use numpy's where function
    index = (human_vertices == skirt_vertex).all(axis=1).nonzero()[0]
    if len(index) > 0:
        # Note that there can be multiple vertices with the same position due to split vertices on edge
        # We are just using the first one found here
        vertex_mapping[i] = index[0]

# Human mesh mask we want to transfer (For example purposes, assuming a simple mask here)
# In reality, this should be loaded or computed according to your specific scenario
human_mask = np.zeros(len(human_mesh.vertices), dtype=int)
for human_index in vertex_mapping.values():
    human_mask[human_index] = 1


# Now map the mask from the human vertices to the skirt vertices using our vertex_mapping
skirts_mask = [
    human_mask[vertex_mapping[i]] if i in vertex_mapping else 0
    for i in range(len(skirts_vertices))
]

# Reconstruct the face information for the skirts mesh using human vertex indices
mapped_faces = []
for face in skirts_mesh.faces:
    mapped_face = [vertex_mapping[vert_index] for vert_index in face]
    mapped_faces.append(mapped_face)

human_mask_np = np.asarray(human_mask)
np.save("short_human_mask.npy", human_mask_np)

skirts_mask_np = np.asarray(skirts_mask)
np.save("short_skirts_mask.npy", skirts_mask_np)

skirts_faces_np = np.asarray(skirts_mesh.faces)
np.save("short_skirts_faces_single.npy", skirts_faces_np)

mapped_faces_np = np.asarray(mapped_faces)
np.save("short_skirts_faces.npy", mapped_faces_np)

# Build new mesh using human vertices but with face connectivity from skirts mesh
new_skirts_mesh = trimesh.Trimesh(vertices=human_vertices, faces=mapped_faces)

# Save the newly created mesh to an OBJ file
new_skirts_mesh.export("new_skirts.obj")
