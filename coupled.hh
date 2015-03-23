#ifndef COUPLED_HH
#define COUPLED_HH
// The material parameters for matrix (PU) are taken from
// http://www.iplex.com.au/iplex.php?page=lib&lib=1&sec=2
// The values for carbon fibres are taken from
// http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp



#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>



#include <fstream>
#include <iostream>

#include "parameters.hh"



namespace Composite_elasticity_problem
{
  using namespace dealii;


  template<int dim> class ElasticProblem;
  class FiberSubproblem;




  class CoupledProblem
  {
  public:
	  CoupledProblem(const std::string &input_file);
	  ~CoupledProblem();

	  void run ();


  private:
	  void setup_system();
	  void solve();

	  BlockSparsityPattern bsp;
	  BlockSparseMatrix<double> bm;
	  BlockVector<double> brhs;

	  ElasticProblem<3> *elastic;
	  FiberSubproblem *fibers;

	  Parameters::AllParameters parameters;

  };




}

#endif
