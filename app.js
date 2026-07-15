// State management
let assets = [];         // Keeps active assets for 2D game loop
let selectedAsset = null;
let allAssets = [];      // Raw assets from API
let groupedAssets = {};  // object_id -> array of style assets
let selectedObjectId = null;
let selectedStyle = null;

// Game Canvas parameters
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
let spriteImg = new Image();
let spriteX = 300;
let spriteY = 200;
let keys = {};
let particles = [];
let soundEnabled = true;

// Three.js 3D parameters
let is3DMode = false;
let scene, camera, renderer, currentModel, orbitControls;
let threeAnimId = null;

// Web Audio API context for zero-dependency sound effects
let audioCtx = null;

// Initialize Audio Context on click (browser security policies)
function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

// Play synthesizer beeps based on action type
function playSound(type) {
    if (!soundEnabled) return;
    initAudio();
    if (!audioCtx) return;

    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        const now = audioCtx.currentTime;

        if (type === 'click') {
            osc.frequency.setValueAtTime(600, now);
            gain.gain.setValueAtTime(0.1, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);
            osc.start(now);
            osc.stop(now + 0.05);
        } else if (type === 'interaction') {
            // Play a cute rising synthetic chime
            osc.frequency.setValueAtTime(440, now);
            osc.frequency.exponentialRampToValueAtTime(880, now + 0.3);
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
            osc.start(now);
            osc.stop(now + 0.3);
        } else if (type === 'movement') {
            // Low feedback rumble
            osc.frequency.setValueAtTime(100, now);
            gain.gain.setValueAtTime(0.05, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
            osc.start(now);
            osc.stop(now + 0.1);
        }
    } catch (e) {
        console.warn("Audio play failed:", e);
    }
}

// Particle System
class Particle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 8;
        this.vy = (Math.random() - 0.5) * 8 - 2; // slight upward drift
        this.alpha = 1.0;
        this.color = color;
        this.size = Math.random() * 6 + 2;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.alpha -= 0.02;
    }

    draw(context) {
        context.save();
        context.globalAlpha = this.alpha;
        context.fillStyle = this.color;
        context.beginPath();
        context.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        context.fill();
        context.restore();
    }
}

function spawnParticles(x, y, actionType) {
    let color = '#7b2cbf';
    let count = 25;
    
    if (actionType.toLowerCase() === 'drink') {
        color = '#00f5d4'; // bright glowing teal
    } else if (actionType.toLowerCase() === 'crafting' || actionType.toLowerCase() === 'repair') {
        color = '#ff9f1c'; // orange sparks
    } else if (actionType.toLowerCase() === 'attack' || actionType.toLowerCase() === 'combat') {
        color = '#ff0054'; // red impact sparks
    }
    
    for (let i = 0; i < count; i++) {
        particles.push(new Particle(x, y, color));
    }
}

// Load dynamic list of generated assets
async function loadAssets() {
    const listContainer = document.getElementById('assets-list');
    listContainer.innerHTML = '<div class="loading-text">Loading catalog...</div>';
    
    try {
        const response = await fetch('/api/assets');
        allAssets = await response.json();
        
        listContainer.innerHTML = '';
        if (allAssets.length === 0) {
            listContainer.innerHTML = '<div class="empty-message">No assets generated yet.<br>Run batch_generator.py first!</div>';
            return;
        }

        // Group raw assets by object_id
        groupedAssets = {};
        allAssets.forEach(asset => {
            const objId = asset.object_id || asset.id;
            if (!groupedAssets[objId]) {
                groupedAssets[objId] = [];
            }
            groupedAssets[objId].push(asset);
        });

        // Populate sidebar with unique game objects
        Object.keys(groupedAssets).forEach(objId => {
            const styles = groupedAssets[objId];
            const primaryAsset = styles[0]; // use first style for name/details
            
            const item = document.createElement('div');
            item.className = 'asset-item';
            item.dataset.id = objId;
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'asset-name';
            nameSpan.textContent = primaryAsset.name;
            
            const typeSpan = document.createElement('span');
            typeSpan.className = 'asset-type-badge';
            typeSpan.textContent = primaryAsset.json ? primaryAsset.json.type : 'Unknown';
            
            item.appendChild(nameSpan);
            item.appendChild(typeSpan);
            
            item.addEventListener('click', () => {
                playSound('click');
                selectedObjectId = objId;
                
                // Keep selected style if available on the new object, else default to first available style
                const availableStyles = styles.map(s => s.style);
                if (!selectedStyle || !availableStyles.includes(selectedStyle)) {
                    selectedStyle = availableStyles[0];
                }
                
                const targetAsset = styles.find(s => s.style === selectedStyle) || primaryAsset;
                selectAsset(targetAsset);
                
                document.querySelectorAll('.asset-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
            });
            
            listContainer.appendChild(item);
        });

        // Auto-select first asset
        if (Object.keys(groupedAssets).length > 0) {
            const firstItem = listContainer.querySelector('.asset-item');
            if (firstItem) firstItem.click();
        }
    } catch (e) {
        console.error("Failed to load assets:", e);
        listContainer.innerHTML = '<div class="empty-message">Error connecting to local server.<br>Ensure server.py is running!</div>';
    }
}

// Select asset and display stats + render in game canvas
function selectAsset(asset) {
    selectedAsset = asset;
    const inspector = document.getElementById('inspector-content');
    inspector.className = 'inspector-content';
    
    if (!asset.has_sprite) {
        inspector.innerHTML = `<div class="empty-message">Asset has no generated sprite image.</div>`;
        return;
    }

    // Initialize active state if stateful
    if (!asset.activeState && asset.json && asset.json.states) {
        asset.activeState = asset.json.current_state || Object.keys(asset.json.states)[0];
    }

    // Extract paths based on active state
    let activeSpriteUrl = asset.sprite_url;
    let activeModelUrl = asset.model_3d_url || (asset.json && asset.json.model_3d);
    let activeDescription = asset.json ? (asset.json.description || "") : "";
    let activeInteractions = asset.json ? (asset.json.interactions || []) : [];

    if (asset.activeState && asset.json && asset.json.states && asset.json.states[asset.activeState]) {
        const stateObj = asset.json.states[asset.activeState];
        activeSpriteUrl = stateObj.sprite_url;
        activeModelUrl = stateObj.model_3d_url || stateObj.model_3d;
        activeDescription = stateObj.description;
        activeInteractions = stateObj.interactions || [];
    }

    // Load sprite image
    spriteImg = new Image();
    spriteImg.src = activeSpriteUrl + '?t=' + new Date().getTime(); // bypass browser cache
    spriteX = 300;
    spriteY = 200;

    // Manage 3D Toggle visibility
    const viewModeBtn = document.getElementById('view-mode-btn');
    if (activeModelUrl) {
        viewModeBtn.style.display = 'inline-block';
        if (is3DMode) {
            load3DModel(activeModelUrl);
        }
    } else {
        viewModeBtn.style.display = 'none';
        if (is3DMode) {
            // Force revert to 2D
            viewModeBtn.click();
        }
    }

    // Render Stats panel
    const data = asset.json || { name: asset.name, type: 'Unknown', hp: 0, speed: 0, interactions: [], hitboxes: { width: 1, height: 1 } };
    
    let interactionsHtml = '';
    if (activeInteractions && activeInteractions.length > 0) {
        interactionsHtml = `
            <div class="actions-panel">
                <h3>Perform Actions</h3>
                <div class="actions-grid">
                    ${activeInteractions.map(act => `<button class="action-btn" data-action="${act}">${act}</button>`).join('')}
                </div>
            </div>
        `;
    }

    // Build style tabs HTML
    const objId = asset.object_id || asset.id;
    const styles = groupedAssets[objId] || [asset];
    let styleSelectorHtml = '';
    if (styles.length > 1) {
        styleSelectorHtml = `
            <div class="style-selector-container">
                <h3>Visual Art Style</h3>
                <div class="style-tabs">
                    ${styles.map(s => {
                        const styleLabel = s.style.replace('-', ' ').toUpperCase();
                        const isActive = s.style === selectedStyle ? 'active' : '';
                        return `<button class="style-tab ${isActive}" data-style="${s.style}">${styleLabel}</button>`;
                    }).join('')}
                </div>
            </div>
        `;
    }

    // Build state badges HTML if stateful
    let statesBadgeHtml = '';
    if (asset.json && asset.json.states) {
        statesBadgeHtml = `
            <div class="state-badges-container" style="margin-bottom: 20px;">
                <h3 style="font-size: 11px; text-transform: uppercase; color: var(--accent-color); margin-bottom: 8px;">Object State</h3>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    ${Object.keys(asset.json.states).map(s => {
                        const isActive = s === asset.activeState;
                        return `<span class="state-badge" style="
                            padding: 4px 10px;
                            border-radius: 4px;
                            font-size: 11px;
                            font-weight: 600;
                            text-transform: uppercase;
                            background: ${isActive ? 'var(--accent-color)' : '#1a1a24'};
                            color: ${isActive ? '#ffffff' : '#a0a0b0'};
                            border: 1px solid ${isActive ? 'var(--accent-color)' : 'var(--border-color)'};
                        ">${s}</span>`;
                    }).join('')}
                </div>
            </div>
        `;
    }

    inspector.innerHTML = `
        <div class="inspector-header">
            <div class="inspector-name">${data.name}</div>
            <div class="inspector-type">${data.type}</div>
        </div>
        
        ${styleSelectorHtml}
        ${statesBadgeHtml}
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Health Points</div>
                <div class="stat-value">${data.hp || 'N/A'}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Speed Stats</div>
                <div class="stat-value">${data.speed !== undefined ? data.speed : 'N/A'}</div>
            </div>
        </div>

        ${activeDescription ? `
        <div style="margin: 15px 0; font-size: 12.5px; color: #a0a0b0; line-height: 1.5; font-style: italic;">
            "${activeDescription}"
        </div>
        ` : ''}

        ${interactionsHtml}

        ${asset.comparison_url && asset.comparison_url !== 'N/A' ? `
        <div class="comparison-inspector" style="margin-top: 20px;">
            <h3 style="margin-bottom: 8px; font-size: 14px; text-transform: uppercase; color: var(--accent-color);">Reference Comparison</h3>
            <div style="overflow-x: auto; width: 100%;">
                <img src="${asset.comparison_url}?t=${new Date().getTime()}" alt="Reference Comparison" style="display: block; max-height: 250px; border-radius: 6px; border: 1px solid var(--border-color); background: #1a1a24; padding: 4px;" />
            </div>
        </div>
        ` : ''}

        <div class="code-inspector">
            <h3>Object Logic JSON</h3>
            <pre><code>${JSON.stringify(data, null, 2)}</code></pre>
        </div>
    `;

    // Add click listeners to interaction actions
    const actionBtns = inspector.querySelectorAll('.action-btn');
    actionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            playSound('interaction');
            spawnParticles(spriteX, spriteY, action);

            // Handle state transitions
            if (asset.json && asset.json.transitions && asset.activeState) {
                const transition = asset.json.transitions.find(t => t.from === asset.activeState && t.trigger.toLowerCase() === action.toLowerCase());
                if (transition) {
                    console.log(`[+] Transitioning state: ${asset.activeState} -> ${transition.to} via ${action}`);
                    asset.activeState = transition.to;
                    setTimeout(() => {
                        selectAsset(asset);
                    }, 250);
                }
            }
        });
    });

    // Add click listeners to style tabs
    const styleTabs = inspector.querySelectorAll('.style-tab');
    styleTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const styleName = tab.dataset.style;
            playSound('click');
            selectedStyle = styleName;
            const targetAsset = styles.find(s => s.style === styleName);
            if (targetAsset) {
                targetAsset.activeState = null; // Reset active state for style switch
                selectAsset(targetAsset);
            }
        });
    });
}

// Three.js 3D Setup & Loaders
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    const useWhiteBg = document.getElementById('toggle-white-bg').checked;
    scene.background = new THREE.Color(useWhiteBg ? 0xffffff : 0x0f0f18);

    // Camera
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 10, 45);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.innerHTML = ''; // clear loading state
    container.appendChild(renderer.domElement);

    // Lighting (Studio style with higher contrast directional lights)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.95);
    dirLight1.position.set(15, 30, 20);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-15, -30, -20);
    scene.add(dirLight2);

    // Controls
    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.05;
    orbitControls.maxPolarAngle = Math.PI / 2; // Don't go below floor
    orbitControls.update();
}

function load3DModel(objUrl) {
    if (!scene) initThree();

    // Clear previous model from scene
    if (currentModel) {
        scene.remove(currentModel);
    }

    const style = (selectedAsset && selectedAsset.json && selectedAsset.json.style) ? selectedAsset.json.style.toLowerCase() : "pixel-art";
    const isPixelated = (style === "pixel-art" || style === "voxels" || style === "isometric-2.5d");
    const isFlat = (style === "pixel-art" || style === "voxels" || style === "low-poly" || style === "isometric-2.5d");
    const hasOutlines = (style === "pixel-art" || style === "voxels" || style === "low-poly" || style === "isometric-2.5d");

    // Load texture map (nearest-neighbor filtering for retro styles, linear filtering for realistic/vector styles)
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(selectedAsset.sprite_url, (tex) => {
        if (isPixelated) {
            tex.magFilter = THREE.NearestFilter;
            tex.minFilter = THREE.NearestFilter;
        } else {
            tex.magFilter = THREE.LinearFilter;
            tex.minFilter = THREE.LinearMipmapLinearFilter;
        }
        tex.needsUpdate = true;
    });

    const material = new THREE.MeshStandardMaterial({
        map: texture,
        transparent: true,
        alphaTest: 0.2,
        side: THREE.DoubleSide,
        roughness: style === "realistic-high-poly" ? 0.5 : 0.8,
        metalness: style === "realistic-high-poly" ? 0.3 : 0.0,
        flatShading: isFlat
    });

    const loader = new THREE.OBJLoader();
    loader.load(objUrl, (obj) => {
        obj.traverse((child) => {
            if (child.isMesh) {
                child.material = material;

                // Add clean outlines only to highlight voxel edges for retro blocky styles
                if (hasOutlines) {
                    const edges = new THREE.EdgesGeometry(child.geometry);
                    const line = new THREE.LineSegments(
                        edges, 
                        new THREE.LineBasicMaterial({ 
                            color: 0x121216, // Dark slate outline for style
                            linewidth: 1 
                        })
                    );
                    child.add(line);
                }
            }
        });

        // Center 3D model inside the viewport bounds
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        obj.position.sub(center);

        // Resize model scaling to fit canvas space
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 22.0 / (maxDim || 1.0);
        obj.scale.set(scale, scale, scale);

        currentModel = obj;
        scene.add(currentModel);

        orbitControls.target.set(0, 0, 0);
        orbitControls.update();
    }, 
    (xhr) => {
        console.log((xhr.loaded / xhr.total * 100) + '% loaded');
    },
    (err) => {
        console.error("3D model loader error:", err);
    });
}

function animateThree() {
    if (!is3DMode) return;
    threeAnimId = requestAnimationFrame(animateThree);
    
    if (currentModel) {
        currentModel.rotation.y += 0.006; // Rotate model slowly
    }

    if (orbitControls) {
        orbitControls.update();
    }

    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// 2D/3D Mode switcher bindings
const viewModeBtn = document.getElementById('view-mode-btn');
viewModeBtn.addEventListener('click', () => {
    is3DMode = !is3DMode;
    
    const canvasEl = document.getElementById('game-canvas');
    const threeEl = document.getElementById('three-container');
    const titleEl = document.getElementById('sandbox-title');
    const instructionsEl = document.getElementById('movement-instructions');
    const gridWrapper = document.getElementById('grid-toggle-wrapper');
    const hitboxWrapper = document.getElementById('hitbox-toggle-wrapper');
    
    if (is3DMode) {
        viewModeBtn.textContent = "Switch to 2D View";
        titleEl.textContent = "3D Voxel Viewer";
        canvasEl.style.display = 'none';
        threeEl.style.display = 'block';
        instructionsEl.style.display = 'none';
        gridWrapper.style.display = 'none';
        hitboxWrapper.style.display = 'none';
        
        if (selectedAsset && selectedAsset.has_model_3d) {
            load3DModel(selectedAsset.model_3d_url);
            animateThree();
        }
    } else {
        viewModeBtn.textContent = "Switch to 3D View";
        titleEl.textContent = "2D Game Sandbox";
        canvasEl.style.display = 'block';
        threeEl.style.display = 'none';
        instructionsEl.style.display = 'flex';
        gridWrapper.style.display = 'flex';
        hitboxWrapper.style.display = 'flex';
        
        if (threeAnimId) {
            cancelAnimationFrame(threeAnimId);
        }
    }
    playSound('click');
});

// WASD Keyboard listeners
window.addEventListener('keydown', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(e.code)) {
        if (document.activeElement === canvas || document.body) {
            keys[e.code] = true;
            initAudio();
        }
    }
});

window.addEventListener('keyup', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(e.code)) {
        keys[e.code] = false;
    }
});

// Focus state handler
canvas.addEventListener('click', () => {
    canvas.focus();
    document.getElementById('focus-badge').textContent = "Keyboard Focused (WASD Active)";
    document.getElementById('focus-badge').style.borderColor = "var(--success-color)";
    document.getElementById('focus-badge').style.color = "var(--success-color)";
});

document.addEventListener('click', (e) => {
    if (e.target !== canvas) {
        document.getElementById('focus-badge').textContent = "Click Sandbox to Move Sprite";
        document.getElementById('focus-badge').style.borderColor = "var(--border-color)";
        document.getElementById('focus-badge').style.color = "var(--text-secondary)";
    }
}, true);

// Main 2D Game Rendering Loop (Canvas)
function draw() {
    requestAnimationFrame(draw);
    
    const useWhiteBg = document.getElementById('toggle-white-bg').checked;
    ctx.fillStyle = useWhiteBg ? '#ffffff' : '#181824';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Grid floor background
    const showGrid = document.getElementById('toggle-grid').checked;
    if (showGrid) {
        ctx.strokeStyle = useWhiteBg ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        const gridSize = 32;
        for (let x = 0; x < canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    }

    if (!selectedAsset || is3DMode) return;

    // Movement speeds integration from JSON
    const speedMultiplier = selectedAsset.json && selectedAsset.json.speed ? selectedAsset.json.speed : 4;
    let moved = false;

    if (keys['KeyW'] || keys['ArrowUp']) { spriteY -= speedMultiplier; moved = true; }
    if (keys['KeyS'] || keys['ArrowDown']) { spriteY += speedMultiplier; moved = true; }
    if (keys['KeyA'] || keys['ArrowLeft']) { spriteX -= speedMultiplier; moved = true; }
    if (keys['KeyD'] || keys['ArrowRight']) { spriteX += speedMultiplier; moved = true; }

    // Sound feedback for movement
    if (moved && Math.random() < 0.15) {
        playSound('movement');
    }

    // Keep sprite in bounds
    if (spriteX < 32) spriteX = 32;
    if (spriteX > canvas.width - 32) spriteX = canvas.width - 32;
    if (spriteY < 32) spriteY = 32;
    if (spriteY > canvas.height - 32) spriteY = canvas.height - 32;

    // Draw generated transparent sprite image
    if (spriteImg.complete) {
        const size = 128; // scale down 1024x1024 to fit canvas
        ctx.drawImage(spriteImg, spriteX - size/2, spriteY - size/2, size, size);
        
        // Draw Hitbox boundaries overlay
        const showHitbox = document.getElementById('toggle-hitbox').checked;
        if (showHitbox && selectedAsset.json && selectedAsset.json.hitboxes) {
            const h = selectedAsset.json.hitboxes;
            const tileWidth = 32;
            const hitW = (h.width || 1) * tileWidth;
            const hitH = (h.height || 1) * tileWidth;
            
            ctx.fillStyle = 'rgba(255, 0, 84, 0.2)';
            ctx.strokeStyle = '#ff0054';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.rect(spriteX - hitW/2, spriteY + (size/4) - hitH, hitW, hitH);
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = '#ff0054';
            ctx.font = '10px Share Tech Mono';
            ctx.fillText(`HITBOX: ${h.width}x${h.height}`, spriteX - hitW/2, spriteY + (size/4) - hitH - 5);
        }
    }

    // Update and draw active particles
    particles.forEach((p, idx) => {
        p.update();
        p.draw(ctx);
        if (p.alpha <= 0) {
            particles.splice(idx, 1);
        }
    });
}

// Initial binding
document.getElementById('refresh-btn').addEventListener('click', loadAssets);
document.getElementById('toggle-sound').addEventListener('change', (e) => {
    soundEnabled = e.target.checked;
});
document.getElementById('toggle-white-bg').addEventListener('change', (e) => {
    if (scene) {
        scene.background = new THREE.Color(e.target.checked ? 0xffffff : 0x0f0f18);
    }
});

// Startup
loadAssets();
draw();
