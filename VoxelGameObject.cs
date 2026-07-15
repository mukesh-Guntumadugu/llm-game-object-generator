using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class HitboxData
{
    public float width;
    public float height;
}

[System.Serializable]
public class GameObjectData
{
    public string name;
    public string type;
    public int hp;
    public float speed;
    public HitboxData hitboxes;
    public List<string> interactions;
    public string sprite_url;
    public string model_3d;
    public string style;
}

public class VoxelGameObject : MonoBehaviour
{
    [Tooltip("Relative or absolute path to the object logic JSON file.")]
    public string jsonFilePath = "Assets/red_apple_object_data.json";

    [Header("Loaded Properties")]
    public string objectName;
    public string objectType;
    public int hp;
    public float speed;
    public List<string> interactions;
    public string artStyle;

    [Header("Collider Settings")]
    public bool autoGenerateCollider = true;

    private GameObjectData data;

    void Start()
    {
        LoadMetadata();
    }

    public void LoadMetadata()
    {
        string fullPath = Path.Combine(Application.dataPath, jsonFilePath.Replace("Assets/", ""));
        
        // Check if file exists, fallback to reading directory directly
        if (!File.Exists(fullPath))
        {
            fullPath = jsonFilePath; // try raw path
        }

        if (File.Exists(fullPath))
        {
            try
            {
                string jsonText = File.ReadAllText(fullPath);
                data = JsonUtility.FromJson<GameObjectData>(jsonText);

                // Populate fields
                objectName = data.name;
                objectType = data.type;
                hp = data.hp;
                speed = data.speed;
                interactions = data.interactions;
                artStyle = data.style;

                Debug.Log($"[VoxelGameObject] Successfully loaded metadata for: {objectName} ({objectType})");

                if (autoGenerateCollider)
                {
                    ConfigureCollider();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[VoxelGameObject] Failed to parse JSON metadata: {e.Message}");
            }
        }
        else
        {
            Debug.LogWarning($"[VoxelGameObject] Metadata JSON file not found at path: {fullPath}");
        }
    }

    private void ConfigureCollider()
    {
        if (data == null || data.hitboxes == null) return;

        // Remove any existing BoxCollider
        BoxCollider oldCollider = GetComponent<BoxCollider>();
        if (oldCollider != null)
        {
            DestroyImmediate(oldCollider);
        }

        // Add fresh BoxCollider based on hitbox values
        BoxCollider boxCollider = gameObject.AddComponent<BoxCollider>();
        
        // Extrusion thickness standard for voxels in Unity
        float depth = 1.0f; 

        // Set size: Width maps to X, Height maps to Y, Voxel Depth maps to Z
        boxCollider.size = new Vector3(data.hitboxes.width, data.hitboxes.height, depth);
        
        // Center the collider on the object mesh
        boxCollider.center = new Vector3(0, data.hitboxes.height / 2.0f, 0);

        Debug.Log($"[VoxelGameObject] Configured BoxCollider with size {boxCollider.size}");
    }
}
