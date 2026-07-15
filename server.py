import os
import json
import http.server
import socketserver

PORT = 8000

class AssetHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # API Endpoint to list generated assets dynamically
        if self.path == '/api/assets':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            assets = []
            assets_dir = 'generated_assets'
            if os.path.exists(assets_dir):
                for style_folder in sorted(os.listdir(assets_dir)):
                    style_path = os.path.join(assets_dir, style_folder)
                    if not os.path.isdir(style_path):
                        continue
                        
                    # Helper function to process a directory for assets
                    def process_directory(dir_path, sub_url_path):
                        dir_files = sorted(os.listdir(dir_path))
                        json_files = [f for f in dir_files if f.endswith('_object_data.json') and os.path.isfile(os.path.join(dir_path, f))]
                        
                        for json_filename in json_files:
                            obj_id = json_filename[:-len('_object_data.json')]
                            json_file = os.path.join(dir_path, json_filename)
                            
                            json_data = None
                            try:
                                with open(json_file, 'r') as f:
                                    json_data = json.load(f)
                            except Exception:
                                continue
                                
                            sprite_file = os.path.join(dir_path, f"{obj_id}_sprite.png")
                            comparison_file = os.path.join(dir_path, f"{obj_id}_comparison.png")
                            model_file = os.path.join(dir_path, f"{obj_id}_model.obj")
                            
                            has_sprite = os.path.exists(sprite_file)
                            if not has_sprite:
                                has_sprite = any(f.startswith(f"{obj_id}_sprite_") and f.endswith('.png') for f in dir_files)
                                
                            has_model_3d = os.path.exists(model_file)
                            if not has_model_3d:
                                has_model_3d = any(f.startswith(f"{obj_id}_model_") and f.endswith('.obj') for f in dir_files)
                                
                            sprite_url = f'{sub_url_path}/{obj_id}_sprite.png'
                            if json_data and 'sprite_url' in json_data:
                                sprite_url = json_data['sprite_url']
                                
                            model_3d_url = f'{sub_url_path}/{obj_id}_model.obj'
                            if json_data and 'model_3d' in json_data:
                                model_3d_url = json_data['model_3d']
                            elif json_data and 'model_3d_url' in json_data:
                                model_3d_url = json_data['model_3d_url']
                                
                            asset_data = {
                                'id': f"{obj_id}_{sub_url_path.replace('/', '_')}",
                                'object_id': obj_id,
                                'style': style_folder,
                                'name': obj_id.replace('_', ' ').title(),
                                'has_json': True,
                                'has_sprite': has_sprite,
                                'has_comparison': os.path.exists(comparison_file),
                                'has_model_3d': has_model_3d,
                                'sprite_url': sprite_url,
                                'json_url': f'{sub_url_path}/{json_filename}',
                                'comparison_url': f'{sub_url_path}/{obj_id}_comparison.png' if os.path.exists(comparison_file) else 'N/A',
                                'model_3d_url': model_3d_url,
                                'json': json_data
                            }
                            assets.append(asset_data)
                            
                    # 1. Process legacy direct files in style_path
                    process_directory(style_path, f'/generated_assets/{style_folder}')
                    
                    # 2. Process new timestamp subfolders
                    for item in sorted(os.listdir(style_path)):
                        item_path = os.path.join(style_path, item)
                        if os.path.isdir(item_path):
                            process_directory(item_path, f'/generated_assets/{style_folder}/{item}')
                        
            self.wfile.write(json.dumps(assets).encode('utf-8'))
        else:
            return super().do_GET()

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), AssetHandler) as httpd:
        print(f"[*] Visual Game Asset Tester running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Shutting down server.")
