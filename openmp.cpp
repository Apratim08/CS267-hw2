#include "common.h"
#include <cmath>
#include <vector>
#include <omp.h>

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

int num_bins_x;
int num_bins_y;
std::vector<std::list<int>> bins;
std::vector<std::list<int>> moveout;
std::vector<std::list<int>> movein;

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    num_bins_x = static_cast<int>(size / cutoff) - 1;
    num_bins_y = static_cast<int>(size / cutoff) - 1;
    bins.resize(num_bins_x*num_bins_y);
    moveout.resize(num_bins_x*num_bins_y);
    movein.resize(num_bins_x*num_bins_y);
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        int bin_x = static_cast<int>(parts[i].x / (size / num_bins_x));
        int bin_y = static_cast<int>(parts[i].y / (size / num_bins_y));
        int bin_index = bin_x + bin_y * num_bins_x;
        bins[bin_index].push_back(i);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    #pragma omp for
    // Compute forces within each bin and neighboring bins
    for (int bin_index = 0; bin_index < num_bins_x * num_bins_y; ++bin_index) {
        int bx = bin_index%num_bins_x;
        int by = bin_index/num_bins_x;

            // Iterate over particles in the current bin
            for (int particle : bins[bin_index]) {
                // Iterate over neighboring bins
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        int nbx = bx + dx;
                        int nby = by + dy;

                        // Check if the neighboring bin is valid
                        if (nbx >= 0 && nbx < num_bins_x && nby >= 0 && nby < num_bins_y) {
                            int neighbor_bin_index = nbx + nby * num_bins_x;
                            // Iterate over particles in the neighboring bin
                            for (int neighbor : bins[neighbor_bin_index]) {
                                apply_force(parts[particle], parts[neighbor]);
                            }
                        }
                    }
                }
            }
    }

    #pragma omp for
    for (int bin_index = 0; bin_index < num_bins_x * num_bins_y; ++bin_index) {
        int bx = bin_index%num_bins_x;
        int by = bin_index/num_bins_x;

        for (int particle : bins[bin_index]) {
            move(parts[particle], size);
            int nbin_x = static_cast<int>(parts[particle].x / (size / num_bins_x));
            int nbin_y = static_cast<int>(parts[particle].y / (size / num_bins_y));
            int new_bin_index = nbin_x + nbin_y * num_bins_x;
            if(nbin_x != bx || nbin_y != by) {
                moveout[bin_index].push_back(particle);
                #pragma omp critical
                movein[new_bin_index].push_back(particle);
            }
            parts[particle].ax = parts[particle].ay = 0;
        }

    }

    #pragma omp for
    for (int bin_index = 0; bin_index < num_bins_x * num_bins_y; ++bin_index) {
        for (int i : moveout[bin_index]) {
            bins[bin_index].remove(i);
        }
        for (int i : movein[bin_index]) {
            bins[bin_index].push_back(i);
        }
        moveout[bin_index].clear();
        movein[bin_index].clear();
    }

}
