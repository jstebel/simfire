#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/transpose_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>
#include <limits>

#include "coupled.hh"
#include "elastic.hh"
#include "fiber.hh"


using namespace Composite_elasticity_problem;
using namespace dealii;



CoupledProblem::CoupledProblem(const std::string &input_file)
	: elastic(nullptr),
	  fibers(nullptr),
	  parameters(input_file)
{}


CoupledProblem::~CoupledProblem()
{
	if (elastic != nullptr) delete elastic;
	if (fibers != nullptr) delete fibers;
}


void CoupledProblem::setup_system()
{
	if (parameters.use_1d_fibers)
	{
		fibers->attach_matrix_handler(elastic->get_dof_handler());

		unsigned int n_3d = elastic->get_dof_handler().n_dofs();
		unsigned int n_1d = fibers->dof_handler().n_dofs();
		unsigned int m_c = fibers->get_constraint_matrix_n_rows();

		bsp.reinit(3,3);
		bsp.block(0,0).reinit(n_3d, n_3d, 200);
		bsp.block(0,1).reinit(n_3d, n_1d, 0);
		bsp.block(0,2).reinit(n_3d, m_c, 80);
		bsp.block(1,0).reinit(n_1d, n_3d, 0);
		bsp.block(1,1).reinit(n_1d, n_1d, 2);
		bsp.block(1,2).reinit(n_1d, m_c, 8);
		bsp.block(2,0).reinit(m_c, n_3d, 16);
		bsp.block(2,1).reinit(m_c, n_1d, 2);
		bsp.block(2,2).reinit(m_c, m_c, 0);
		bsp.collect_sizes();

		DoFTools::make_sparsity_pattern (elastic->get_dof_handler(), bsp.block(0,0));
		fibers->set_sparsity_pattern(bsp.block(1,1));
		fibers->set_constraint_matrix_sparsity_pattern(bsp.block(0,2), bsp.block(1,2), bsp.block(2,0), bsp.block(2,1));
		bsp.compress();

		bm.reinit(bsp);

		brhs.reinit({n_3d, n_1d, m_c});
	}

	elastic->setup_system();
}


void CoupledProblem::solve ()
{
  SparseDirectUMFPACK umf;

  if (parameters.use_1d_fibers)
  {
	  umf.solve(bm, brhs);
	  elastic->set_solution(brhs.block(0));
	  fibers->set_solution(brhs.block(1));
  }
  else
  {
	  umf.solve(bm.block(0,0), brhs.block(0));
	  elastic->set_solution(brhs.block(0));
  }
}


void CoupledProblem::run ()
{
	elastic = new ElasticProblem<3>(parameters);

	if (parameters.use_1d_fibers)
		fibers = new FiberSubproblem(parameters.mesh1d_file);

	setup_system();

	elastic->assemble_system(bm.block(0,0), brhs.block(0));

	if (parameters.use_1d_fibers)
	{
		fibers->assemble_fiber_matrix(bm.block(1,1), parameters.Young_modulus_fiber, parameters.Fiber_volume_ratio);
		fibers->assemble_constraint_mat(bm.block(0,2), bm.block(1,2), bm.block(2,0), bm.block(2,1));
	}

//	std::ofstream f("matrix.dat");
//	bm.block(2,1).print_pattern(f);
//	f.close();

	solve();

	elastic->output_results ();
	fibers->output_results(parameters.output_file_base);
}






