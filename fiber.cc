#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <fstream>

#include "fiber.hh"


using namespace Composite_elasticity_problem;
using namespace dealii;













FiberSubproblem::FiberSubproblem(const std::string &mesh_file)
	: fe(1),
	  dh(tri)

{
	GridIn<1,3> grid_in;
	grid_in.attach_triangulation(tri);

	std::ifstream input_file(mesh_file);
	Assert (input_file, ExcFileNotOpen(mesh_file.c_str()));

	grid_in.read_msh(input_file);

	std::cout << "   Number of fiber active cells:       "
			<< tri.n_active_cells()
			<< std::endl;


	dh.distribute_dofs(fe);
}

FiberSubproblem::~FiberSubproblem()
{
}

void FiberSubproblem::attach_matrix_handler(const DoFHandler<3> &dh_)
{
	dh_3d = &dh_;

	MappingQ1<3> map;
	for (auto node : tri.get_vertices())
	{
		node_to_cell.push_back(GridTools::find_active_cell_around_point(map, *dh_3d, node));
	}
}

unsigned int FiberSubproblem::get_constraint_matrix_n_rows()
{
	return 1*tri.n_lines();
}

void FiberSubproblem::set_sparsity_pattern(SparsityPattern &sp)
{
	CompressedSparsityPattern c_sparsity(dh.n_dofs());
	DoFTools::make_sparsity_pattern(dh, c_sparsity);
	sp.copy_from(c_sparsity);
}

void FiberSubproblem::set_constraint_matrix_sparsity_pattern(SparsityPattern &sp0t, SparsityPattern &sp1t, SparsityPattern &sp0, SparsityPattern &sp1)
{
	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), dof_indices(fe.dofs_per_cell);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		// allocate space for 3d dofs adajcent to line vertices
		for (unsigned int vid=0; vid<2; vid++)
		{
			auto cell = node_to_cell[line->vertex_index(vid)].first;
			cell->get_dof_indices(dof_indices_3d);
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
				sp0.add(id, dof_indices_3d[i]);
				sp0t.add(dof_indices_3d[i], id);
			}
		}

		// allocate space for 1d dofs
		line->get_dof_indices(dof_indices);
		for (unsigned int i=0; i<fe.dofs_per_cell; i++)
		{
			sp1.add(id,   dof_indices[i]);
			sp1t.add(dof_indices[i], id);
		}
	}
}



void FiberSubproblem::assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1)
{
	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), dof_indices(fe.dofs_per_cell);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		line->get_dof_indices(dof_indices);
		Point<3> tangent = line->vertex(1)-line->vertex(0);
		tangent /= tangent.norm();

		for (unsigned int vid=0; vid<2; vid++)
		{
			auto cell = node_to_cell[line->vertex_index(vid)].first;
			Quadrature<3> q(node_to_cell[line->vertex_index(vid)].second);
			FEValues<3> fe_values(*fe_3d, q, update_values);
			
			fe_values.reinit(cell);
			cell->get_dof_indices(dof_indices_3d);
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
				cm0.add(id, dof_indices_3d[i],
						fe_values.shape_value(i,0)*tangent[fe_3d->system_to_component_index(i).first]);
				cm0t.add(dof_indices_3d[i], id,
						fe_values.shape_value(i,0)*tangent[fe_3d->system_to_component_index(i).first]);
			}

			cm1.add(id, dof_indices[vid], -1);
			cm1t.add(dof_indices[vid], id, -1);
		}
	}
}




void FiberSubproblem::assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, double E_fiber, double fiber_volume)
{
	QGauss<1> quadrature_formula(2);
	FEValues<1,3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();
	FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<1,3>::active_cell_iterator cell = dh.begin_active(),
			end_cell = dh.end();
	for (; cell!=end_cell; ++cell)
	{
		fe_values.reinit(cell);
		cell_matrix = 0;
		cell_rhs = 0;
		for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
		{
			for (unsigned int i=0; i<dofs_per_cell; i++)
				for (unsigned int j=0; j<dofs_per_cell; j++)
					cell_matrix(i,j) += fiber_volume*E_fiber*(fe_values.shape_grad (i, q_index) *
							fe_values.shape_grad (j, q_index) *
							fe_values.JxW (q_index));

//			 for (unsigned int i=0; i<dofs_per_cell; ++i)
//				 cell_rhs(i) += (fe_values.shape_value (i, q_index) *
//						 1 * fe_values.JxW (q_index));
		}

		cell->get_dof_indices (local_dof_indices);
		for (unsigned int i=0; i<dofs_per_cell; ++i)
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				fiber_matrix.add (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));

//		for (unsigned int i=0; i<dofs_per_cell; ++i)
//			fiber_rhs(local_dof_indices[i]) += cell_rhs(i);
	}

//	std::map<types::global_dof_index,double> boundary_values;
//	VectorTools::interpolate_boundary_values (dh,
//			5,
//			ZeroFunction<3>(),
//			boundary_values);
//	MatrixTools::apply_boundary_values (boundary_values,
//			fiber_matrix,
//			fiber_solution,
//			fiber_rhs);
//
//	fiber_matrix.set(0,0,0);

}





void FiberSubproblem::output_results(const std::string &base_path, const Vector<double> &solution_3d) const
{
	std::string filename = base_path + "-fiber-displacement.vtk";
	std::ofstream output (filename.c_str());

	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), local_dof_indices(dofs_per_cell);
	FESystem<1,3> fe_disp(FE_Q<1,3>(1),3);
	DoFHandler<1,3> dh_disp(tri);
	dh_disp.distribute_dofs(fe_disp);
	DataOut<1,DoFHandler<1,3> > data_out;
	Vector<double> dx(dh.n_dofs()), dy(dh.n_dofs()), dz(dh.n_dofs());
	std::vector<unsigned int> count(dh.n_dofs(), 0.);

	data_out.attach_dof_handler(dh);

	DoFHandler<1,3>::active_cell_iterator cell = dh.begin_active(),
			end_cell = dh.end();
	for (; cell!=end_cell; ++cell)
	{
		cell->get_dof_indices (local_dof_indices);
		Point<3> tangent = cell->vertex(1) - cell->vertex(0);
		tangent /= tangent.norm();
		for (int vid=0; vid<2; vid++)
		{
			auto cell3d = node_to_cell[cell->vertex_index(vid)].first;
			Quadrature<3> q(node_to_cell[cell->vertex_index(vid)].second);
			FEValues<3> fe_values(*fe_3d, q, update_values);

			fe_values.reinit(cell3d);
			cell3d->get_dof_indices(dof_indices_3d);
			Point<3> disp_3d;
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
				disp_3d[fe_3d->system_to_component_index(i).first] += fe_values.shape_value(i,0)*solution_3d[dof_indices_3d[i]];
			}

			disp_3d -= (disp_3d*tangent)*tangent;
			dx[local_dof_indices[vid]] += disp_3d[0];
			dy[local_dof_indices[vid]] += disp_3d[1];
			dz[local_dof_indices[vid]] += disp_3d[2];


			dx[local_dof_indices[vid]] += tangent[0]*fiber_solution[local_dof_indices[vid]];
			dy[local_dof_indices[vid]] += tangent[1]*fiber_solution[local_dof_indices[vid]];
			dz[local_dof_indices[vid]] += tangent[2]*fiber_solution[local_dof_indices[vid]];
			count[local_dof_indices[vid]]++;
		}
	}
	for (unsigned int i=0; i<dx.size(); i++)
	{
		dx[i] /= count[i];
		dy[i] /= count[i];
		dz[i] /= count[i];
	}

//	std::vector<std::string> solution_names;
//	std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
//	for (int i=0; i<3; i++)
//	{
//		solution_names.push_back ("displacement");
//		interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
//	}

//	data_out.add_data_vector (displacement,
//			solution_names,
//			DataOut<1, DoFHandler<1,3> >::type_automatic,
//			interpretation);

	data_out.add_data_vector (fiber_solution, "solution");
	data_out.add_data_vector (dx, "dx");
	data_out.add_data_vector (dy, "dy");
	data_out.add_data_vector (dz, "dz");
	data_out.build_patches ();
	data_out.write_vtk (output);
}







