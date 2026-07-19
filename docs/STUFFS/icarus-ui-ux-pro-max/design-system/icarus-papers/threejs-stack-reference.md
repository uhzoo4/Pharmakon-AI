## UI Pro Max Stack Guidelines
**Stack:** threejs | **Query:** mesh material blending transparency performance
**Source:** stacks/threejs.csv | **Found:** 4 results

### Result 1
- **Category:** Performance
- **Guideline:** InstancedMesh for Repeated Objects
- **Description:** Use THREE.InstancedMesh when rendering 50 or more identical objects. It submits all N transforms in one draw call instead of N draw calls and reduces CPU-GPU communication overhead dramatically.
- **Do:** Use InstancedMesh for any group of 50+ meshes sharing the same geometry and material
- **Don't:** Create 50+ separate Mesh objects with the same geometry and material
- **Code Good:** const COUNT = 500; const iMesh = new THREE.InstancedMesh(geo, mat, COUNT); const matrix = new THREE.Matrix4(); for (let i = 0; i < COUNT; i++) { matrix.setPosition(Math.random()*10, Math.random()*10, Math.random()*10); iMesh.setMatrixAt(i, matrix); } iMesh.instanceMatrix.needsUpdate = true; scene.ad...
- **Code Bad:** for (let i = 0; i < 500; i++) { scene.add(new THREE.Mesh(geo, mat)); } // 500 separate draw calls per frame
- **Severity:** High
- **Docs URL:** https://threejs.org/docs/#api/en/objects/InstancedMesh

### Result 2
- **Category:** GSAP
- **Guideline:** Tween Three.js Properties Directly
- **Description:** GSAP can tween any numeric JavaScript property including mesh.position.x mesh.rotation.y and material.opacity. No wrapper or adaptor is needed. Note: to tween material.opacity the material must have transparent:true set before the tween starts.
- **Do:** Pass Three.js object properties directly to gsap.to(); set transparent:true before tweening opacity
- **Don't:** Use a plain proxy object then manually copy values to Three.js properties every frame
- **Code Good:** gsap.to(mesh.rotation, { y: Math.PI * 2, duration: 2, ease: 'power1.inOut' }); mesh.material.transparent = true; // required before tweening opacity gsap.to(mesh.material, { opacity: 0, duration: 1 });
- **Code Bad:** const tw = { v: 0 }; gsap.to(tw, { v: Math.PI * 2, onUpdate: () => mesh.rotation.y = tw.v }); // unnecessary proxy wrapper
- **Severity:** Medium
- **Docs URL:** https://gsap.com/docs/v3/GSAP/gsap.to()

### Result 3
- **Category:** Geometry
- **Guideline:** Share Geometry Across Meshes
- **Description:** When multiple objects share the same shape create one geometry instance and pass it to every Mesh. Each Mesh gets its own transform and material while all share a single GPU buffer.
- **Do:** Create one geometry and pass the same reference to every Mesh constructor
- **Don't:** Create a separate identical geometry inside a loop for each object
- **Code Good:** const geo = new THREE.BoxGeometry(1, 1, 1); // one GPU buffer for (let i = 0; i < 200; i++) { const m = new THREE.Mesh(geo, mat); m.position.set(Math.random() * 10, 0, Math.random() * 10); scene.add(m); }
- **Code Bad:** for (let i = 0; i < 200; i++) { const geo = new THREE.BoxGeometry(1, 1, 1); // 200 separate GPU buffers scene.add(new THREE.Mesh(geo, mat)); }
- **Severity:** Critical
- **Docs URL:** https://threejs.org/docs/#api/en/core/BufferGeometry

### Result 4
- **Category:** Particles
- **Guideline:** BufferGeometry Plus Points for Particle Systems
- **Description:** Build all particle systems with BufferGeometry plus a Float32Array position attribute rendered as Points. Never use individual Mesh objects as particles — they cannot scale past a few hundred with good performance.
- **Do:** Use Points plus BufferGeometry for all particle effects
- **Don't:** Create hundreds of individual Mesh objects to simulate a particle system
- **Code Good:** const COUNT = 3000; const geo = new THREE.BufferGeometry(); const pos = new Float32Array(COUNT * 3); for (let i = 0; i < COUNT * 3; i++) pos[i] = (Math.random() - 0.5) * 20; geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); const particles = new THREE.Points(geo, new THREE.PointsMater...
- **Code Bad:** for (let i = 0; i < 500; i++) { scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 8), mat)); } // 500 separate draw calls per frame
- **Severity:** High
- **Docs URL:** https://threejs.org/docs/#api/en/objects/Points

