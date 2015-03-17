#ifndef FIBER_HH
#define FIBER_HH



#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <iostream>


namespace Composite_elasticity_problem
{
  using namespace dealii;


  class FiberSubproblem
  {
  public:
	  FiberSubproblem(const std::string &mesh_file);
	  ~FiberSubproblem();

	  void attach_matrix_handler(const DoFHandler<3> &dh_);

	  void add_sparsity_pattern(SparsityPattern &pattern);

	  void modify_stiffness_matrix(SparseMatrix<double> &mat, double E_matrix, double E_fiber, double fiber_volume);

	  void assemble_fiber_matrix(double E_fiber, double fiber_volume);

	  void output_results() const;

  private:

	  Triangulation<1,3> tri;

	  SparsityPattern sp;
	  SparseMatrix<double> fiber_matrix;

	  FE_Q<1,3> fe;
	  DoFHandler<1,3> dh;
	  Vector<double> fiber_solution;
	  Vector<double> fiber_rhs;

	  const DoFHandler<3> *dh_3d;

	  std::vector<std::pair<DoFHandler<3>::active_cell_iterator,Point<3> > > node_to_cell;

	  double E_longitudal;
	  double E_transverse;


  };







}

#endif
