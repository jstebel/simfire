#ifndef FIBER_HH
#define FIBER_HH



#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <iostream>

#include "parameters.hh"


namespace Composite_elasticity_problem
{
  using namespace dealii;


  /**
   * Abstract class FiberSubproblem.
   */
  class FiberSubproblem
  {
  public:
	  FiberSubproblem() {};
	  virtual ~FiberSubproblem() {};

	  virtual void attach_matrix_handler(const DoFHandler<3> &dh_) = 0;

	  virtual void assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1) = 0;

	  virtual void assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, Vector<double> &fiber_rhs, double E_fiber, double nu_fiber, double fiber_volume) = 0;

	  virtual void output_results(const std::string &base_path, const Vector<double> &solution_3d, const Vector<double> &solution_1d) const = 0;

	  virtual DoFHandler<1,3> &dof_handler() = 0;

	  virtual unsigned int get_constraint_matrix_n_rows() = 0;

	  virtual void set_sparsity_pattern(SparsityPattern &sp) = 0;

	  virtual void set_constraint_matrix_sparsity_pattern(SparsityPattern&, SparsityPattern&, SparsityPattern&, SparsityPattern&) = 0;
  };


}

#endif
