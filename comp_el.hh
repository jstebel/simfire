#ifndef COMP_EL_HH
#define COMP_EL_HH
// The material parameters for matrix (PU) are taken from
// http://www.iplex.com.au/iplex.php?page=lib&lib=1&sec=2
// The values for carbon fibres are taken from
// http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>



#include <fstream>
#include <iostream>

#include "parameters.hh"
#include "fiber.hh"

namespace Composite_elasticity_problem
{
  using namespace dealii;






  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem (const std::string &input_file);
    ~ElasticProblem ();
    void run ();

  private:
    void setup_system ();
    Tensor<4,dim> elastic_tensor(unsigned int material_id) const;
    void allocate();
    void assemble_system(SparseMatrix<double> &system_matrix, Vector<double> &system_rhs);
    void solve ();
    void output_results () const;
    void output_stress() const;
    void output_ranges() const;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    FESystem<dim>        fe;

    BlockSparsityPattern bsp;
    BlockSparseMatrix<double> bm;
    BlockVector<double> brhs;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    FiberSubproblem *fibers;

    Parameters::AllParameters parameters;
  };










}

#endif
