import bpy
import sys
import os

def bake_vertex_colors(obj_path, albedo_path):
    # Ensure absolute paths because Blender's relative paths break without a saved .blend file
    obj_path = os.path.abspath(obj_path)
    albedo_path = os.path.abspath(albedo_path)
    
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import OBJ (Blender 3.2+ native OBJ importer reads vertex colors)
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Smart UV Project
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.01) # 66 degrees
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Setup Material for Baking
    mat = bpy.data.materials.new(name="BakeMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create Vertex Color node (Attribute node in Cycles)
    attr_node = nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name = "Color"
    
    # Emission node for flat color baking
    emission_node = nodes.new(type="ShaderNodeEmission")
    
    # Output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    
    # Link Vertex Color -> Emission -> Output
    links.new(attr_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    
    # Create Image Texture node for Bake Target
    img = bpy.data.images.new("BakeTarget", width=1024, height=1024)
    img_node = nodes.new(type="ShaderNodeTexImage")
    img_node.image = img
    
    # Make Image Texture node active for baking
    nodes.active = img_node
    img_node.select = True
    
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
        
    # Setup Cycles for Baking
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 1
    
    # Bake Emit pass
    bpy.ops.object.bake(type='EMIT')
    
    # Save Image
    img.filepath = albedo_path
    img.file_format = 'PNG'
    img.save()
    
    # Remove material, re-export OBJ with UVs (strip vertex colors to keep it clean)
    obj.data.materials.clear()
    
    # Export UV mapped OBJ
    bpy.ops.wm.obj_export(filepath=obj_path, export_materials=False)
    
if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] # get args after "--"
    obj_path = argv[0]
    albedo_path = argv[1]
    bake_vertex_colors(obj_path, albedo_path)
