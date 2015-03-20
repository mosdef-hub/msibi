//
//  structure_matching.cpp
//  
//
//  Created by Christopher Iacovella on 2/3/13.
//
//

//#include "/Users/cri/Projects/init_code/repository/group-code/common.hpp"
#include "/home/tcmoore3/group-code/common.hpp"
#include "/home/tcmoore3/group-code/structure_matching.hpp"



int main()
{

    double rdf_cutoff = 5.0;
    double pot_cutoff = 3.0;
    double dr = 0.05;
    double start = 0.85;
    int smooth_radius = 1;

    std::vector<double> alpha;
    alpha.push_back(0.5);
    alpha.push_back(0.7);
    alpha.push_back(0.5);
    
		// vector to hold target data and information for each state point
    std::vector<state_point> states;
    state_point state_temp;
    states.push_back(state_temp);
    states.push_back(state_temp);
    states.push_back(state_temp);


		// load target and set up query script for each state point
    int ss = 0;
    states[ss].set_box(12.0);   // cubic box 12x12x12
    states[ss].set_temperature(0.5, 1.0);   //pass T and kb to set units
    states[ss].set_target_traj("../target0/target.dcd", F_DCD);   // load target trajectory
    states[ss].set_query_traj("q0/query.dcd", F_DCD);   // tell it where to put the query traj
    states[ss].set_query_script("q0/template.txt", "q0/query.hoomd.txt");   // sets T, dt, etc...
    states[ss].set_coord_prototype(1, "../prototype/coord_prototype.txt");   // loads types, first number is the number of atoms in the molecule
    
    ss++;
    states[ss].set_box(13.0);
    states[ss].set_temperature(1.5, 1.0); //pass T and kb to set units
    states[ss].set_target_traj("../target2/target.dcd", F_DCD);
    states[ss].set_query_traj("q2/query.dcd", F_DCD);
    states[ss].set_query_script("q2/template.txt", "q2/query.hoomd.txt");
    states[ss].set_coord_prototype(1, "../prototype/coord_prototype.txt");

    ss++;
    states[ss].set_box(20.0);
    states[ss].set_temperature(2.0, 1.0); //pass T and kb to set units
    states[ss].set_target_traj("../target3/target.dcd", F_DCD);
    states[ss].set_query_traj("q3/query.dcd", F_DCD);
    states[ss].set_query_script("q3/template.txt", "q3/query.hoomd.txt");
    states[ss].set_coord_prototype(1, "../prototype/coord_prototype.txt");

    
		// vector to hold the potentials
    std::vector<intermolecular> potentials;
    intermolecular pot_temp;
    potentials.push_back(pot_temp);

  
    potentials[0].set_pair(1,1);   // only 1 type, between types 1 and 1
    potentials[0].init_statepoints(states, start, dr, rdf_cutoff, pot_cutoff);
    potentials[0].init_potential_PMF(states, 1.0, 1.0, 0.9);   // first potential guess from PMF from RDF

		// we should output the target RDF and need to write out the initial potential
    for(int i=0; i<potentials.size(); i++)
    {
        potentials[i].write_target_RDF();
        potentials[i].write_potential_hoomd();
    }
    
    generate_hoomd_header(states, potentials);
    //optimization loop
    std::ofstream errfile("error.dat");
    for(int t=0;t<=200; t++)
    {
			// run the query simulations
        system("bash runner.sh");
        std::cout << "======="<< std::endl;
        std::cout << "iteration:\t" << t << std::endl;
        std::cout << "======="<< std::endl;

        for(int i=0; i<potentials.size(); i++)
        {

            potentials[i].calc_query_rdf(states);

            //output the potential and RDF every 5 timesteps
            if(t%1 == 0)
            {
                potentials[i].write_query_RDF(t);
                potentials[i].write_potential_full(t);
            }

            bool smooth = false;
            if(t%200 == 0)
                smooth = true;
            double error = 0; 
            potentials[i].tweak_potential(states, alpha, ALPHA_LINEAR, smooth, smooth_radius, error);  // ALPHA_LINEAR lets us say apply the linear scaling of alpha.
	    errfile << t << " " << error << std::endl;
        }
    }
   
    return 0;
}


