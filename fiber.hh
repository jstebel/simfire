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

	  void allocate_constraint_mat();

	  void assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1);

	  void assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, double E_fiber, double fiber_volume);

	  void output_results(const std::string &base_path) const;

	  SparseMatrix<double> &get_fiber_matrix() { return fiber_matrix; }

	  SparseMatrix<double> &get_constraint_matrix(unsigned int part, bool trans=false) { return trans?constraint_mat_t[part]:constraint_mat[part]; }

	  Vector<double> &get_fiber_rhs() { return fiber_rhs; }

	  SparsityPattern &get_sparsity_pattern() { return sp; }

	  SparsityPattern &get_constraint_matrix_sparsity_pattern(unsigned int part, bool trans=false) { return trans?sp_cm_t[part]:sp_cm[part]; }

	  void set_solution(Vector<double> &sol) { fiber_solution = sol; }

	  DoFHandler<1,3> &dof_handler() { return dh; }

	  unsigned int get_constraint_matrix_n_rows();

	  void set_sparsity_pattern(SparsityPattern &sp);

	  void set_constraint_matrix_sparsity_pattern(SparsityPattern&, SparsityPattern&, SparsityPattern&, SparsityPattern&);

  private:

	  Triangulation<1,3> tri;
	  FE_Q<1,3> fe;
	  DoFHandler<1,3> dh;

	  SparsityPattern sp;
	  SparseMatrix<double> fiber_matrix;
	  SparsityPattern sp_cm[2];
	  SparsityPattern sp_cm_t[2];
	  SparseMatrix<double> constraint_mat[2];
	  SparseMatrix<double> constraint_mat_t[2];

	  Vector<double> fiber_solution;
	  Vector<double> fiber_rhs;

	  const DoFHandler<3> *dh_3d;

	  std::vector<std::pair<DoFHandler<3>::active_cell_iterator,Point<3> > > node_to_cell;

	  double E_longitudal;
	  double E_transverse;


  };







}

#endif
