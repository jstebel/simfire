#ifndef FIBER1D_HH
#define FIBER1D_HH



#include "fiber.hh"


namespace Composite_elasticity_problem
{
  using namespace dealii;


  /**
   * Fiber model with response only for stretching.
   */
  class Fiber1D : public FiberSubproblem
  {
  public:
	  Fiber1D(const std::string &mesh_file);
	  ~Fiber1D() override;

	  void attach_matrix_handler(const DoFHandler<3> &dh_) override;

	  void assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1) override;

	  void assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, Vector<double> &fiber_rhs, double E_fiber, double nu_fiber, double fiber_volume) override;

	  void output_results(const std::string &base_path, const Vector<double> &solution_3d, const Vector<double> &solution_1d) const override;

	  DoFHandler<1,3> &dof_handler()  override { return dh; }

	  unsigned int get_constraint_matrix_n_rows() override;

	  void set_sparsity_pattern(SparsityPattern &sp) override;

	  void set_constraint_matrix_sparsity_pattern(SparsityPattern&, SparsityPattern&, SparsityPattern&, SparsityPattern&) override;

  private:

	  Triangulation<1,3> tri;
	  FE_Q<1,3> fe;
	  DoFHandler<1,3> dh;
	  const DoFHandler<3> *dh_3d;

	  std::vector<std::pair<DoFHandler<3>::active_cell_iterator,Point<3> > > node_to_cell;

  };



}

#endif
