#include "/Users/tcmoore3/group-code/common.hpp"

/*
 * calculate M for the potential as a function of time
 * decide which value 
 * 
 */


/********************************************************************/
	// SOME FUNCTIONS FER US
/********************************************************************/
double LJ(double r, double epsilon, double sigma)
{
	double V = 0.0;
	if(r <= 0)
		V  = 99999.2;
	else
		V = 4*epsilon*(pow((sigma/r), 12.0)-pow((sigma/r), 6.0));
	return V;
}
double M_sum(const coordlist_t &U, const int &dim, const double &r_min, const double &r_max)
{
	double sum_U;
	for(int i = 0; i < U.size(); i++)
	{
		double r = U[i][0];
		if(r >= r_min  && r <= r_max)
		{
			sum_U += fabs(U[i][dim]);
		}
	}
	return sum_U;
}

double M(const coordlist_t &test, const coordlist_t &target, const double &r_min, const double &r_max)
{
		// make sure the "r" values are the same in each case
		// by making sure they are the same size with same inital and final values
	assert(test.size() == target.size());
	assert(test[0][0] == target[0][0]);
	assert(test[test.size()-1][0] == target[target.size()-1][0]);
	double M = 0;
	for(int i = 0; i < test.size(); i++)
	{
		double r = test[i][0];
		if(r >= r_min  &&  r <= r_max)
			M += fabs(test[i][1] - target[i][1]);
	}
	double sum = 0;
	sum += M_sum(test, 1, r_min, r_max);
	sum += M_sum(target, 1, r_min, r_max);
	M /= sum;
	return M;
}

/********************************************************************/





int main()
{
		// read data into arrays
	coordlist_t target_rdf0, target_rdf1, target_rdf2;
	std::vector<coordlist_t> target_potential;
	std::ifstream rdf_t0("rdfs/rdf.target0.t1t1.txt");
	std::ifstream rdf_t1("rdfs/rdf.target1.t1t1.txt");
	std::ifstream rdf_t2("rdfs/rdf.target2.t1t1.txt");
	std::string line;
	std::ofstream err0("error0.dat");
	std::ofstream err1("error1.dat");
	std::ofstream err2("error2.dat");
	std::ofstream pot("error_pot.dat");
	while(std::getline(rdf_t0, line))
	{
		double r, gr;
		std::stringstream(line) >> r >> gr;
		coord_t point;
		point.push_back(r);
		point.push_back(gr);
		target_rdf0.push_back(point);
	}
		//std::cout << target_rdf0.size() << std::endl;
	while(std::getline(rdf_t1, line))
	{
		double r, gr;
		std::stringstream(line) >> r >> gr;
		coord_t point;
		point.push_back(r);
		point.push_back(gr);
		target_rdf1.push_back(point);
	}
	while(std::getline(rdf_t2, line))
	{
		double r, gr;
		std::stringstream(line) >> r >> gr;
		coord_t point;
		point.push_back(r);
		point.push_back(gr);
		target_rdf2.push_back(point);
	}
	
	double sigma = 1.0;
	double epsilon = 1.0;
	int iter = 130/5 + 1;
	int skip = 1;
	int max_iter = 200;
	for(int i = 0; i <= max_iter; i+=skip)
	{
		char s0 [50];
		char s1 [50];
		char s2 [50];
		char s3 [50];
		sprintf(s0, "rdfs/rdf.%d.query0.t1t1.txt", i);
		sprintf(s1, "rdfs/rdf.%d.query1.t1t1.txt", i);
		sprintf(s2, "rdfs/rdf.%d.query2.t1t1.txt", i);
		sprintf(s3, "potentials/pot_full.time%d.t1t1.txt", i);
		std::ifstream rdf_test0(s0);
		std::ifstream rdf_test1(s1);
		std::ifstream rdf_test2(s2);
		std::ifstream pot_test(s3);
		coordlist_t v_test_rdf0, v_test_rdf1, v_test_rdf2, test_pot;
		while(std::getline(rdf_test0, line))
		{
			double r, gr;
			std::stringstream(line) >> r >> gr;
			coord_t point;
			point.push_back(r);
			point.push_back(gr);
			v_test_rdf0.push_back(point);
		}
			//std::cout << v_test_rdf0.size() << std::endl;
		while(std::getline(rdf_test1, line))
		{
			double r, gr;
			std::stringstream(line) >> r >> gr;
			coord_t point;
			point.push_back(r);
			point.push_back(gr);
			v_test_rdf1.push_back(point);
		}
		while(std::getline(rdf_test2, line))
		{
			double r, gr;
			std::stringstream(line) >> r >> gr;
			coord_t point;
			point.push_back(r);
			point.push_back(gr);
			v_test_rdf2.push_back(point);
		}
		while(std::getline(pot_test, line))
		{
			double r, gr;
			std::stringstream(line) >> r >> gr;
			coord_t point;
			point.push_back(r);
			point.push_back(gr);
			test_pot.push_back(point);
		}
		coordlist_t pot_target = test_pot;
		for(int j = 0; j < pot_target.size(); j++)
		{
			pot_target[j][1] = LJ(pot_target[j][0], epsilon, sigma);
				//std::cout << pot_target[j][0] << " " << LJ(pot_target[j][0], epsilon, sigma) << std::endl;
		}
			// now just send each vector to a function that returns M
			//std::cout << i << " " << 1.0 - M(test_pot, pot_target, 1.1, 2.5) << std::endl;
		coord_t errors;
		std::cout << target_rdf0.size() << " " << v_test_rdf0.size() << std::endl;
		std::cout << target_rdf1.size() << " " << v_test_rdf1.size() << std::endl;
		std::cout << target_rdf2.size() << " " << v_test_rdf2.size() << std::endl;
		errors.push_back(1.0 - M(target_rdf0, v_test_rdf0, 0.0, 5.0));
		errors.push_back(1.0 - M(target_rdf1, v_test_rdf1, 0.0, 5.0));
		errors.push_back(1.0 - M(target_rdf2, v_test_rdf2, 0.0, 5.0));
		errors.push_back(1.0 - M(test_pot, pot_target, 1.1, 2.5));
		err0 << i << " " << errors[0] << std::endl;
		err1 << i << " " << errors[1] << std::endl;
		err2 << i << " " << errors[2] << std::endl;
		pot << i << " " << errors[3] << std::endl;
		std::cout << "Done did " << i << std::endl;
	}
	return 0;
	
}
