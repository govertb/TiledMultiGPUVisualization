/*
 ==============================================================================

 RPBarnesHutApproximator.cpp
 Copyright © 2016, 2017, 2018  G. Brinkmann

 This file is part of graph_viewer.

 graph_viewer is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 graph_viewer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with graph_viewer.  If not, see <https://www.gnu.org/licenses/>.

 ==============================================================================
*/

#include "RPBarnesHutApproximator.hpp"
#include <math.h>
#include <stdlib.h>
#include <queue>

namespace RPGraph
{
    BarnesHutCell::BarnesHutCell(Coordinate position, float length, Coordinate particle_position, float particle_mass)
    : cell_center{position}, length{length}, mass_center(particle_position), total_mass{particle_mass}
    {
        lb = position.x - length/2.0;
        rb = position.x + length/2.0;
        bb = position.y - length/2.0;
        ub = position.y + length/2.0;
    }

    BarnesHutCell::~BarnesHutCell()
    {
        for (nid_t n = 0; n < 4; ++n) delete sub_cells[n];
    }

    void BarnesHutCell::add_leafcell(int quadrant, float mass, Coordinate pos)
    {
        Coordinate leafcell_center_coordinate = Coordinate(0,0);
        if (quadrant == 0)
            leafcell_center_coordinate = Coordinate(this->cell_center.x-length/4.0,this->cell_center.y+length/4);
        else if (quadrant == 1)
            leafcell_center_coordinate = Coordinate(this->cell_center.x+length/4.0,this->cell_center.y+length/4);
        else if (quadrant == 2)
            leafcell_center_coordinate = Coordinate(this->cell_center.x+length/4.0,this->cell_center.y-length/4);
        else if (quadrant == 3)
            leafcell_center_coordinate = Coordinate(this->cell_center.x-length/4.0,this->cell_center.y-length/4);

        sub_cells[quadrant] = new BarnesHutCell(leafcell_center_coordinate, this->length/2.0, pos, mass);
        num_subparticles += 1;

    }

    BarnesHutApproximator::BarnesHutApproximator(Coordinate root_center, float root_length, float theta)
    : root_center{root_center}, root_length{root_length}, theta{theta}
    {
        this->reset(root_center, root_length);
    }

    void BarnesHutApproximator::reset(Coordinate root_center, float root_length)
    {
        delete root_cell; // this recursively deletes the entire tree
        root_cell = nullptr;

        this->root_center = root_center;
        this->root_length = root_length;
    }


    Real2DVector BarnesHutApproximator::approximateForce(Coordinate particle_pos, float particle_mass, float theta)
    {
        Real2DVector force = Real2DVector(0.0, 0.0);
        std::queue<BarnesHutCell*> cells_to_check;
        cells_to_check.push(root_cell);

        BarnesHutCell *cur_cell;
        while (!cells_to_check.empty())
        {
            cur_cell = cells_to_check.front();
            cells_to_check.pop();

            const float D2 = distance2(particle_pos, cur_cell->mass_center);
            if (D2 == 0)
            {
                // If we approximate the force of a particle on itself...
                if (cur_cell->num_subparticles == 0) continue;
                else return Real2DVector(rand(), rand());

            }

            // length / D >= theta is the criterion to divide into subcells.
            if (cur_cell->length*cur_cell->length / D2 < theta*theta || cur_cell->num_subparticles == 0)
                force += direction(particle_pos, cur_cell->mass_center)  *
                (particle_mass * cur_cell->total_mass / D2);

            else
                for (int i = 0; i < 4; ++i)
                    if (cur_cell->sub_cells[i] != nullptr) cells_to_check.push(cur_cell->sub_cells[i]);
        }
        return force;
    }

    void BarnesHutApproximator::insertParticle(RPGraph::Coordinate particle_position, float particle_mass)
    {
        if(not root_cell)
        {
            root_cell = new BarnesHutCell(this->root_center, this->root_length,
                                          particle_position, particle_mass);
        }

        else
        {
            BarnesHutCell *cur_cell = root_cell;
            while (true)
            {
                const int quadrant_new_particle = (particle_position-cur_cell->cell_center).quadrant();

                if (particle_position.y > cur_cell->ub or particle_position.x > cur_cell->rb or
                    particle_position.x < cur_cell->lb or particle_position.y < cur_cell->bb)
                {
                    //fprintf(stderr, "error: Barnes-Hut: Can't insert particle out of bounds of this cell.\n");
                    return;
                }

                // N.B. a BarnesHutCell is never empty, but can lack subparticles/cells.
                // If so, we need to create, and insert, a subcell for the single particle that
                // is stored in this cell.
                if (cur_cell->num_subparticles == 0)
                {
                    if (particle_position == cur_cell->mass_center)
                    {
                        // We want two particles in the same place...
                        // Thats equivalent to a single particle with summed masses.
                        // mass_center won't change.
                        cur_cell->total_mass += particle_mass;
                        return;
                    }

                    // We move the single particle to a subcell.
                    int quadrant_existing_particle = (cur_cell->mass_center - cur_cell->cell_center).quadrant();
                    cur_cell->add_leafcell(quadrant_existing_particle, cur_cell->total_mass, cur_cell->mass_center);
                }

                // We assume inserting will succeed, and update total_mass and mass_center accordingly
                cur_cell->total_mass  += particle_mass;
                cur_cell->mass_center  = cur_cell->mass_center * (float) (cur_cell->num_subparticles);
                cur_cell->mass_center += particle_position;
                cur_cell->mass_center /= (cur_cell->num_subparticles+1);

                // If we can add a leaf-cell in an empty slot, we do so.
                if (cur_cell->sub_cells[quadrant_new_particle] == nullptr)
                {
                    cur_cell->add_leafcell(quadrant_new_particle, particle_mass, particle_position);
                    return;
                }

                // Else we recurse to the occupied cell.
                else
                {
                    cur_cell->num_subparticles += 1;
                    cur_cell = cur_cell->sub_cells[quadrant_new_particle];
                }
            }
        }
    }
}
