#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <fstream>

#include "fiber.hh"


using namespace Composite_elasticity_problem;
using namespace dealii;













FiberSubproblem::FiberSubproblem(const std::string &mesh_file)
{
	GridIn<1,3> grid_in;
	grid_in.attach_triangulation(tri);

	std::ifstream input_file(mesh_file);
	Assert (input_file, ExcFileNotOpen(mesh_file.c_str()));

	grid_in.read_msh(input_file);

	std::cout << "   Number of fiber active cells:       "
			<< tri.n_active_cells()
			<< std::endl;

}

FiberSubproblem::~FiberSubproblem()
{
}

void FiberSubproblem::attach_matrix_handler(const DoFHandler<3> &dh_)
{
	dh = &dh_;

	MappingQ1<3> map;

	for (auto node : tri.get_vertices())
		node_to_cell.push_back(GridTools::find_active_cell_around_point(map, *dh, node));

}


void FiberSubproblem::add_sparsity_pattern(SparsityPattern &pattern)
{
	const FiniteElement<3> *fe = &dh->get_fe();
	const int dofs_per_cell = fe->dofs_per_cell;

	std::vector<types::global_dof_index> dof_indices0 (dofs_per_cell),
			dof_indices1 (dofs_per_cell);

	Triangulation<1,3>::active_cell_iterator line = tri.begin_active(),
			end_line = tri.end();
	for (; line!=end_line; ++line)
	{
		auto cell0 = node_to_cell[line->vertex_index(0)].first;
		auto cell1 = node_to_cell[line->vertex_index(1)].first;

		cell0->get_dof_indices(dof_indices0);
		cell1->get_dof_indices(dof_indices1);

		for (int i=0; i<dofs_per_cell; i++)
			for (int j=0; j<dofs_per_cell; j++)
				if (fe->system_to_component_index(i).first == fe->system_to_component_index(j).first)
				{
					pattern.add(dof_indices0[i],dof_indices0[i]);
					pattern.add(dof_indices0[i],dof_indices1[j]);
					pattern.add(dof_indices1[j],dof_indices0[i]);
					pattern.add(dof_indices1[j],dof_indices1[j]);
				}
	}

}

void FiberSubproblem::modify_stiffness_matrix(SparseMatrix<double> &mat, double E_matrix, double E_fiber, double fiber_volume)
{
	const FiniteElement<3> *fe = &dh->get_fe();
//	const FiniteElement<3> *fe_base = &fe->base_element(0);
	const int dofs_per_cell = fe->dofs_per_cell;

	double shape_sum0, shape_sum1;
	std::vector<double> shape_weight0(dofs_per_cell), shape_weight1(dofs_per_cell);

	std::vector<types::global_dof_index> dof_indices0 (dofs_per_cell),
			dof_indices1 (dofs_per_cell);

	Triangulation<1,3>::active_cell_iterator line = tri.begin_active(),
			end_line = tri.end();
	for (; line!=end_line; ++line)
	{
		auto cell0 = node_to_cell[line->vertex_index(0)].first;
		auto point0 = node_to_cell[line->vertex_index(0)].second;

		auto cell1 = node_to_cell[line->vertex_index(1)].first;
		auto point1 = node_to_cell[line->vertex_index(1)].second;

		cell0->get_dof_indices(dof_indices0);
		cell1->get_dof_indices(dof_indices1);

		Quadrature<3> q0(point0);
		FEValues<3> fe_values0(*fe, q0, update_values);
		fe_values0.reinit(cell0);
		shape_sum0 = 0;
		for (int i=0; i<dofs_per_cell; i++)
		{
			shape_weight0[i] = fe_values0.shape_value(i,0);
			if (fe->system_to_component_index(i).first == 0)
				shape_sum0 += shape_weight0[i];
		}
		for (int i=0; i<dofs_per_cell; i++)
			shape_weight0[i] /= shape_sum0;

		Quadrature<3> q1(point1);
		FEValues<3> fe_values1(*fe, q1, update_values);
		fe_values1.reinit(cell1);
		shape_sum1 = 0;
		for (int i=0; i<dofs_per_cell; i++)
		{
			shape_weight1[i] = fe_values1.shape_value(i,0);
			if (fe->system_to_component_index(i).first == 0)
				shape_sum1 += shape_weight1[i];
		}
		for (int i=0; i<dofs_per_cell; i++)
			shape_weight1[i] /= shape_sum1;

		Point<3> tangent = (line->vertex(1)-line->vertex(0));
		double dx = tangent.norm();
		tangent /= dx;

		for (int i=0; i<dofs_per_cell; i++)
		{
			int ci = fe->system_to_component_index(i).first;
			int bi = fe->system_to_component_index(i).second;

			const double stiff = fiber_volume*(E_fiber - E_matrix)*fabs(tangent[ci])/dx;

			for (int j=0; j<dofs_per_cell; j++)
			{
				int cj = fe->system_to_component_index(j).first;
				int bj = fe->system_to_component_index(j).second;

				if (ci != cj) continue;

				mat.add(dof_indices0[i], dof_indices0[i],  shape_weight0[bi]*shape_weight0[bi]*stiff);
				mat.add(dof_indices0[i], dof_indices1[j], -shape_weight0[bi]*shape_weight1[bj]*stiff);
				mat.add(dof_indices1[j], dof_indices0[i], -shape_weight1[bj]*shape_weight0[bi]*stiff);
				mat.add(dof_indices1[j], dof_indices1[j],  shape_weight1[bj]*shape_weight1[bj]*stiff);
			}
		}



	}
}














