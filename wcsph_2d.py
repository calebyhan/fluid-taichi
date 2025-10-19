import taichi as ti
import numpy as np

ti.init(arch=ti.gpu if ti.cuda else ti.cpu, default_fp=ti.f32)

# Domain
DOMAIN = ti.Vector([2.0, 1.0])
MAX_DT = 0.001
MIN_DT = 0.0001

# Particles - NOW WE CAN HANDLE MORE!
N_PARTICLES = 2500  # Increased from 900
PARTICLE_RADIUS = 0.008
SPACING = 0.016

# Spatial hashing grid
GRID_SIZE = 0.06  # Slightly larger than H for efficiency
GRID_WIDTH = int(DOMAIN[0] / GRID_SIZE) + 1
GRID_HEIGHT = int(DOMAIN[1] / GRID_SIZE) + 1
MAX_NEIGHBORS = 100  # Max neighbors per particle

# SPH Parameters
H = 0.05           # Smoothing radius
MASS = 0.0005
RHO0 = 1000.0      # Water density
STIFFNESS = 200.0
STIFFNESS_NEAR = 1000.0  # For double density relaxation
EXPONENT = 4.0
VISCOSITY = 0.08
SURFACE_TENSION = 0.02
COHESION = 0.01    # Additional surface cohesion
XSPH_FACTOR = 0.5  # XSPH velocity smoothing

# Taichi fields
pos = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
vel_xsph = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
acc = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
density = ti.field(dtype=ti.f32, shape=N_PARTICLES)
density_near = ti.field(dtype=ti.f32, shape=N_PARTICLES)
pressure = ti.field(dtype=ti.f32, shape=N_PARTICLES)
pressure_near = ti.field(dtype=ti.f32, shape=N_PARTICLES)
color_field = ti.field(dtype=ti.f32, shape=N_PARTICLES)
normal = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)

# Spatial hashing
grid_count = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))
grid_particles = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT, MAX_NEIGHBORS))

# Adaptive timestep tracking
max_velocity = ti.field(dtype=ti.f32, shape=())
max_acceleration = ti.field(dtype=ti.f32, shape=())

# Physics
GRAVITY = ti.Vector([0.0, -9.8])
DAMPING = 0.998

# Kernel constants
POLY6_CONST = 315.0 / (64.0 * 3.14159265 * (H ** 9))
SPIKY_GRAD_CONST = -45.0 / (3.14159265 * (H ** 6))
VISCOSITY_LAP_CONST = 45.0 / (3.14159265 * (H ** 6))

@ti.func
def get_cell(pos: ti.types.vector(2, ti.f32)) -> ti.types.vector(2, ti.i32):
    """Convert position to grid cell coordinates"""
    return ti.Vector([
        ti.cast(pos[0] / GRID_SIZE, ti.i32),
        ti.cast(pos[1] / GRID_SIZE, ti.i32)
    ])

@ti.func
def is_valid_cell(cell: ti.types.vector(2, ti.i32)) -> ti.i32:
    """Check if cell is within grid bounds"""
    return 0 <= cell[0] and cell[0] < GRID_WIDTH and 0 <= cell[1] and cell[1] < GRID_HEIGHT

@ti.func
def poly6_kernel(r: ti.f32) -> ti.f32:
    result = 0.0
    if 0.0 <= r and r <= H:
        x = H * H - r * r
        result = POLY6_CONST * x * x * x
    return result

@ti.func
def poly6_gradient(r_vec: ti.types.vector(2, ti.f32)) -> ti.types.vector(2, ti.f32):
    result = ti.Vector([0.0, 0.0])
    r = r_vec.norm()
    if 0.0 < r and r <= H:
        x = H * H - r * r
        g = -6.0 * POLY6_CONST * x * x
        result = g * r_vec
    return result

@ti.func
def spiky_gradient(r_vec: ti.types.vector(2, ti.f32)) -> ti.types.vector(2, ti.f32):
    result = ti.Vector([0.0, 0.0])
    r = r_vec.norm()
    if 0.0 < r and r <= H:
        x = H - r
        g = SPIKY_GRAD_CONST * x * x
        result = g * r_vec / r
    return result

@ti.func
def viscosity_laplacian(r: ti.f32) -> ti.f32:
    result = 0.0
    if 0.0 < r and r <= H:
        result = VISCOSITY_LAP_CONST * (H - r)
    return result

@ti.kernel
def init_particles():
    # Create particles in a dam-break configuration
    particles_per_row = 50
    n_rows = N_PARTICLES // particles_per_row
    
    for i in range(N_PARTICLES):
        row = i // particles_per_row
        col = i % particles_per_row
        
        # Stack particles in bottom-left corner (dam break)
        pos[i] = ti.Vector([
            0.1 + col * SPACING + ti.random() * 0.0005,
            0.1 + row * SPACING + ti.random() * 0.0005
        ])
        vel[i] = ti.Vector([0.0, 0.0])
        vel_xsph[i] = ti.Vector([0.0, 0.0])
        acc[i] = ti.Vector([0.0, 0.0])
        density[i] = RHO0
        density_near[i] = 0.0
        pressure[i] = 0.0
        pressure_near[i] = 0.0

@ti.kernel
def build_spatial_hash():
    """Build spatial hash grid for neighbor finding"""
    # Clear grid
    for i, j in ti.ndrange(GRID_WIDTH, GRID_HEIGHT):
        grid_count[i, j] = 0
    
    # Add particles to grid
    for i in range(N_PARTICLES):
        cell = get_cell(pos[i])
        if is_valid_cell(cell):
            idx = ti.atomic_add(grid_count[cell], 1)
            if idx < MAX_NEIGHBORS:
                grid_particles[cell[0], cell[1], idx] = i

@ti.kernel
def compute_density_double():
    """Double density relaxation (Clavet et al. 2005)"""
    for i in range(N_PARTICLES):
        rho = 0.0
        rho_near = 0.0
        
        cell = get_cell(pos[i])
        
        # Check 3x3 neighborhood
        for offset_x, offset_y in ti.ndrange((-1, 2), (-1, 2)):
            neighbor_cell = cell + ti.Vector([offset_x, offset_y])
            if is_valid_cell(neighbor_cell):
                n_neighbors = grid_count[neighbor_cell]
                for k in range(n_neighbors):
                    j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]
                    if j != i:
                        r_vec = pos[i] - pos[j]
                        r = r_vec.norm()
                        if r <= H:
                            q = r / H
                            if q < 1.0:
                                rho += (1.0 - q) ** 2
                                rho_near += (1.0 - q) ** 3
        
        density[i] = rho
        density_near[i] = rho_near
        
        # Compute pressures
        pressure[i] = STIFFNESS * (rho - RHO0 / MASS)
        pressure_near[i] = STIFFNESS_NEAR * rho_near

@ti.kernel
def compute_color_field():
    """Compute color field gradient for surface tension"""
    for i in range(N_PARTICLES):
        c_sum = 0.0
        n_sum = ti.Vector([0.0, 0.0])
        
        cell = get_cell(pos[i])
        
        # Check 3x3 neighborhood
        for offset_x, offset_y in ti.ndrange((-1, 2), (-1, 2)):
            neighbor_cell = cell + ti.Vector([offset_x, offset_y])
            if is_valid_cell(neighbor_cell):
                n_neighbors = grid_count[neighbor_cell]
                for k in range(n_neighbors):
                    j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]
                    if j != i:
                        r_vec = pos[i] - pos[j]
                        r = r_vec.norm()
                        if r <= H:
                            c_sum += poly6_kernel(r)
                            n_sum += poly6_gradient(r_vec)
        
        color_field[i] = c_sum
        normal[i] = n_sum

@ti.kernel
def compute_forces():
    """Compute all forces using spatial hashing"""
    for i in range(N_PARTICLES):
        force = GRAVITY * MASS
        vel_sum = ti.Vector([0.0, 0.0])
        neighbor_count = 0.0
        
        cell = get_cell(pos[i])
        
        # Check 3x3 neighborhood
        for offset_x, offset_y in ti.ndrange((-1, 2), (-1, 2)):
            neighbor_cell = cell + ti.Vector([offset_x, offset_y])
            if is_valid_cell(neighbor_cell):
                n_neighbors = grid_count[neighbor_cell]
                for k in range(n_neighbors):
                    j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]
                    if j != i:
                        r_vec = pos[i] - pos[j]
                        r = r_vec.norm()
                        if r <= H and r > 0.001:
                            # Pressure force (double density)
                            q = r / H
                            if q < 1.0:
                                # Normalized direction
                                dir = r_vec / r
                                
                                # Pressure
                                p_term = pressure[i] * (1.0 - q) + pressure_near[i] * (1.0 - q) ** 2
                                force -= p_term * dir * 0.5
                                
                                # Viscosity
                                vel_diff = vel[j] - vel[i]
                                visc = VISCOSITY * viscosity_laplacian(r)
                                force += MASS * vel_diff * visc
                                
                                # Cohesion (surface attraction)
                                if r > H * 0.5:
                                    cohesion_force = COHESION * (1.0 - q)
                                    force -= cohesion_force * dir
                                
                                # Accumulate for XSPH
                                vel_sum += vel[j]
                                neighbor_count += 1.0
        
        # Surface tension
        n_norm = normal[i].norm()
        if n_norm > 0.01:
            force += -SURFACE_TENSION * n_norm * normal[i]
        
        acc[i] = force / MASS
        
        # XSPH velocity smoothing
        if neighbor_count > 0:
            vel_xsph[i] = vel[i] + XSPH_FACTOR * (vel_sum / neighbor_count - vel[i])
        else:
            vel_xsph[i] = vel[i]

@ti.kernel
def find_max_values():
    """Find maximum velocity and acceleration for adaptive timestep"""
    max_v = 0.0
    max_a = 0.0
    
    for i in range(N_PARTICLES):
        v = vel[i].norm()
        a = acc[i].norm()
        max_v = ti.max(max_v, v)
        max_a = ti.max(max_a, a)
    
    max_velocity[None] = max_v
    max_acceleration[None] = max_a

@ti.kernel
def integrate(dt: ti.f32):
    """Symplectic Euler integration with XSPH"""
    for i in range(N_PARTICLES):
        # Use XSPH velocity for position update
        vel[i] += acc[i] * dt
        vel[i] *= DAMPING
        
        # Clamp velocity
        v_norm = vel[i].norm()
        max_vel = 15.0
        if v_norm > max_vel:
            vel[i] = vel[i] / v_norm * max_vel
        
        # Update position with smoothed velocity
        pos[i] += vel_xsph[i] * dt

@ti.kernel
def handle_boundaries():
    """Boundary conditions with restitution and friction"""
    restitution = 0.3
    friction = 0.2
    
    for i in range(N_PARTICLES):
        # Bottom
        if pos[i].y < PARTICLE_RADIUS:
            pos[i].y = PARTICLE_RADIUS
            vel[i].y = ti.abs(vel[i].y) * restitution
            vel[i].x *= (1.0 - friction)
        
        # Top
        if pos[i].y > DOMAIN[1] - PARTICLE_RADIUS:
            pos[i].y = DOMAIN[1] - PARTICLE_RADIUS
            vel[i].y = -ti.abs(vel[i].y) * restitution
            vel[i].x *= (1.0 - friction)
        
        # Left
        if pos[i].x < PARTICLE_RADIUS:
            pos[i].x = PARTICLE_RADIUS
            vel[i].x = ti.abs(vel[i].x) * restitution
            vel[i].y *= (1.0 - friction)
        
        # Right
        if pos[i].x > DOMAIN[0] - PARTICLE_RADIUS:
            pos[i].x = DOMAIN[0] - PARTICLE_RADIUS
            vel[i].x = -ti.abs(vel[i].x) * restitution
            vel[i].y *= (1.0 - friction)

def compute_adaptive_timestep():
    """Compute timestep based on CFL condition"""
    find_max_values()
    
    v_max = max_velocity[None]
    a_max = max_acceleration[None]
    
    # CFL condition: particles shouldn't move more than a fraction of H
    dt_cfl = 0.25 * H / max(v_max, 0.01)
    
    # Acceleration condition
    dt_acc = ti.sqrt(H / max(a_max, 0.01))
    
    # Take minimum and clamp
    dt = min(dt_cfl, dt_acc)
    dt = max(MIN_DT, min(dt, MAX_DT))
    
    return dt

def main():
    init_particles()
    gui = ti.GUI('Advanced SPH Fluid Simulator', res=(800, 400), background_color=0x000510)
    
    frame = 0
    time = 0.0
    
    print("Controls:")
    print("  Press SPACE to pause/resume")
    print("  Press R to reset")
    
    paused = False
    
    while gui.running:
        # Handle input
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == 'r':
                init_particles()
                time = 0.0
                frame = 0
        
        if not paused:
            # Adaptive timestep
            dt = compute_adaptive_timestep()
            
            # Simulation steps
            for _ in range(2):
                build_spatial_hash()
                compute_density_double()
                compute_color_field()
                compute_forces()
                integrate(dt)
                handle_boundaries()
            
            time += dt * 2
        
        # Visualization
        p = pos.to_numpy()
        screen_pos = np.column_stack([
            p[:, 0] / DOMAIN[0],
            p[:, 1] / DOMAIN[1]
        ])
        
        # Color by velocity magnitude (blue -> cyan -> white -> yellow -> red)
        velocities = vel.to_numpy()
        speeds = np.linalg.norm(velocities, axis=1)
        speed_max = max(np.max(speeds), 0.01)
        speed_norm = np.clip(speeds / (speed_max * 0.5), 0.0, 1.0)
        
        colors = np.zeros(N_PARTICLES, dtype=np.uint32)
        for i in range(N_PARTICLES):
            t = speed_norm[i]
            if t < 0.5:
                # Blue to cyan to white
                s = t * 2.0
                r = int(s * 255)
                g = int(128 + s * 127)
                b = 255
            else:
                # White to yellow to red
                s = (t - 0.5) * 2.0
                r = 255
                g = int(255 * (1.0 - s))
                b = int(255 * (1.0 - s))
            colors[i] = (r << 16) | (g << 8) | b
        
        # Debug info
        if frame % 60 == 0:
            densities = density.to_numpy()
            pressures = pressure.to_numpy()
            print(f"Frame {frame} | Time: {time:.2f}s | dt: {dt*1000:.3f}ms")
            print(f"  Density:  avg={np.mean(densities):.1f}, max={np.max(densities):.1f}")
            print(f"  Pressure: avg={np.mean(pressures):.1f}, max={np.max(pressures):.1f}")
            print(f"  Speed:    avg={np.mean(speeds):.3f}, max={np.max(speeds):.3f}")
        
        # Render
        gui.circles(screen_pos, radius=3, color=colors)
        
        # UI overlay
        gui.text(f"Frame: {frame} | Time: {time:.2f}s", pos=(0.01, 0.97), color=0xFFFFFF)
        gui.text(f"Particles: {N_PARTICLES} | dt: {dt*1000:.2f}ms", pos=(0.01, 0.93), color=0xFFFFFF)
        if paused:
            gui.text("PAUSED (SPACE to resume)", pos=(0.35, 0.5), color=0xFF0000, font_size=20)
        
        frame += 1
        gui.show()

if __name__ == '__main__':
    main()