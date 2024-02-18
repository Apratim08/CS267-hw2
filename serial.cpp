#include "common.h"
#include <cmath>
#include <vector>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
    // Updating the acceleration of the particles (x &y direction) based on the calculated force
    // components (coeff)
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

const int num_bins_x = 9;
const int num_bins_y = 9;

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Move particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }

    // Clear forces and reassign particles to bins
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        parts[i].bin_x = static_cast<int>(parts[i].x / (size / num_bins_x));
        parts[i].bin_y = static_cast<int>(parts[i].y / (size / num_bins_y));
    }

    // Vector of vectors to store particles in each bin
    std::vector<std::vector<particle_t>> bins(num_bins_x * num_bins_y);

    // Populate bins
    for (int i = 0; i < num_parts; ++i) {
        int bin_index = parts[i].bin_x + parts[i].bin_y * num_bins_x;
         bins[bin_index].emplace_back(parts[i]);
    }

    // Compute forces within each bin and neighboring bins
    for (int bx = 0; bx < num_bins_x; ++bx) {
        for (int by = 0; by < num_bins_y; ++by) {
            int bin_index = bx + by * num_bins_x;

            // Iterate over particles in the current bin
            for (particle_t& particle : bins[bin_index]) {
                // Iterate over neighboring bins
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        // if (dx != 0 || dy != 0) { // Skip the current bin - improved the time by
                        // several seconds
                        int nbx = bx + dx;
                        int nby = by + dy;

                        // Check if the neighboring bin is valid
                        if (nbx >= 0 && nbx < num_bins_x && nby >= 0 && nby < num_bins_y) {
                            int neighbor_bin_index = nbx + nby * num_bins_x;

                            // Iterate over particles in the neighboring bin
                            for (particle_t& neighbor : bins[neighbor_bin_index]) {
                                apply_force(particle, neighbor);
                            }
                        }
                    }
                }
            }
        }
    }
}
