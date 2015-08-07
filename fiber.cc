#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
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













Fiber1D::Fiber1D(const std::string &mesh_file)
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

Fiber1D::~Fiber1D()
{
}

void Fiber1D::attach_matrix_handler(const DoFHandler<3> &dh_)
{
	dh_3d = &dh_;

	MappingQ1<3> map;
	for (auto node : tri.get_vertices())
	{
		node_to_cell.push_back(GridTools::find_active_cell_around_point(map, *dh_3d, node));
	}
}

unsigned int Fiber1D::get_constraint_matrix_n_rows()
{
	return 1*tri.n_lines();
}

void Fiber1D::set_sparsity_pattern(SparsityPattern &sp)
{
	CompressedSparsityPattern c_sparsity(dh.n_dofs());
	DoFTools::make_sparsity_pattern(dh, c_sparsity);
	sp.copy_from(c_sparsity);
}

void Fiber1D::set_constraint_matrix_sparsity_pattern(SparsityPattern &sp0t, SparsityPattern &sp1t, SparsityPattern &sp0, SparsityPattern &sp1)
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



void Fiber1D::assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1)
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




void Fiber1D::assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, Vector<double> &fiber_rhs, double E_fiber, double nu_fiber, double fiber_volume)
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





void Fiber1D::output_results(const std::string &base_path, const Vector<double> &solution_3d, const Vector<double> &solution_1d) const
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


			dx[local_dof_indices[vid]] += tangent[0]*solution_1d[local_dof_indices[vid]];
			dy[local_dof_indices[vid]] += tangent[1]*solution_1d[local_dof_indices[vid]];
			dz[local_dof_indices[vid]] += tangent[2]*solution_1d[local_dof_indices[vid]];
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

	data_out.add_data_vector (solution_1d, "solution");
	data_out.add_data_vector (dx, "dx");
	data_out.add_data_vector (dy, "dy");
	data_out.add_data_vector (dz, "dz");
	data_out.build_patches ();
	data_out.write_vtk (output);
}






































FiberLinear::FiberLinear(const std::string &mesh_file)
	: fe(FE_Q<1,3>(2), 3/*, FE_Q<1,3>(1), 3*/),
	  dh(tri),
	  q_constraint(QGauss<1>(2))

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

FiberLinear::~FiberLinear()
{
}

void FiberLinear::attach_matrix_handler(const DoFHandler<3> &dh_)
{
	dh_3d = &dh_;

	MappingQ1<3> map;
	FEValues<1,3> fe_values(fe, q_constraint, update_quadrature_points);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		fe_values.reinit(line);
		for (unsigned int k=0; k<q_constraint.size(); k++)
			node_to_cell.push_back(GridTools::find_active_cell_around_point(map, *dh_3d, fe_values.quadrature_point(k)));
	}
}

unsigned int FiberLinear::get_constraint_matrix_n_rows()
{
	return 3*tri.n_lines();
}

void FiberLinear::set_sparsity_pattern(SparsityPattern &sp)
{
	CompressedSparsityPattern c_sparsity(dh.n_dofs());
	DoFTools::make_sparsity_pattern(dh, c_sparsity);
	sp.copy_from(c_sparsity);
}

void FiberLinear::set_constraint_matrix_sparsity_pattern(SparsityPattern &sp0t, SparsityPattern &sp1t, SparsityPattern &sp0, SparsityPattern &sp1)
{
	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), dof_indices(fe.dofs_per_cell);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		// allocate space for 3d dofs adajcent to line vertices
		for (unsigned int k=0; k<q_constraint.size(); k++)
		{
			auto cell = node_to_cell[q_constraint.size()*id+k].first;
			cell->get_dof_indices(dof_indices_3d);
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
//				if (fe_3d->system_to_component_index(i).first > 1) continue;
				sp0.add(3*id+fe_3d->system_to_component_index(i).first, dof_indices_3d[i]);
				sp0t.add(dof_indices_3d[i], 3*id+fe_3d->system_to_component_index(i).first);
			}
		}

		// allocate space for 1d dofs
		line->get_dof_indices(dof_indices);
		for (unsigned int i=0; i<fe.dofs_per_cell; i++)
		{
			// consider only displacements in 1d (ignore rotations)
			if (fe.system_to_component_index(i).first < 3)
			{
				sp1.add(3*id+fe.system_to_component_index(i).first,   dof_indices[i]);
				sp1t.add(dof_indices[i], 3*id+fe.system_to_component_index(i).first);
			}
		}
	}
}



void FiberLinear::assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1)
{
	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), dof_indices(fe.dofs_per_cell);
	FEValues<1,3> fe_values1d(fe, q_constraint, update_values);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		line->get_dof_indices(dof_indices);
		Point<3> tangent = line->vertex(1)-line->vertex(0);
		tangent /= tangent.norm();

		fe_values1d.reinit(line);

		for (unsigned int k=0; k<q_constraint.size(); k++)
		{
			auto cell = node_to_cell[q_constraint.size()*id+k].first;
			Quadrature<3> q(node_to_cell[q_constraint.size()*id+k].second);
			FEValues<3> fe_values3d(*fe_3d, q, update_values);

			fe_values3d.reinit(cell);
			cell->get_dof_indices(dof_indices_3d);
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
//				if (fe_3d->system_to_component_index(i).first > 1) continue;
				cm0.add(3*id+fe_3d->system_to_component_index(i).first, dof_indices_3d[i],
						fe_values3d.shape_value(i,0));
				cm0t.add(dof_indices_3d[i], 3*id+fe_3d->system_to_component_index(i).first,
						fe_values3d.shape_value(i,0));
			}

			for (unsigned int i=0; i<fe.dofs_per_cell; i++)
			{
				// consider only displacements in 1d (ignore rotations)
				if (fe.system_to_component_index(i).first < 3)
				{
					cm1.add(3*id+fe.system_to_component_index(i).first, dof_indices[i], -fe_values1d.shape_value(i,k));
					cm1t.add(dof_indices[i], 3*id+fe.system_to_component_index(i).first, -fe_values1d.shape_value(i,k));
				}
			}
		}
	}
}




void FiberLinear::assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, Vector<double> &fiber_rhs, double E_fiber, double nu_fiber, double fiber_volume)
{
	QGauss<1> quadrature_formula(2);
	FEValues<1,3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();
	FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	double G_fiber = 0.5 * E_fiber / (nu_fiber + 1);

	DoFHandler<1,3>::active_cell_iterator cell = dh.begin_active(),
			end_cell = dh.end();
	for (; cell!=end_cell; ++cell)
	{
		fe_values.reinit(cell);
		cell_matrix = 0;
		cell_rhs = 0;
		Point<3> tangent = cell->vertex(1) - cell->vertex(0);
		tangent /= tangent.norm();
		Tensor<2,3> normal_proj;
		for (unsigned int i=0; i<3; i++)
			for (unsigned int j=0; j<3; j++)
				normal_proj[{i,j}] = (i==j?1:0) - tangent[i]*tangent[j];

		for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
		{
			for (unsigned int i=0; i<dofs_per_cell; i++)
				for (unsigned int j=0; j<dofs_per_cell; j++)
					if (fe.system_to_component_index(i).first == fe.system_to_component_index(j).first)
						cell_matrix(i,j) += fiber_volume
								* (E_fiber*(fe_values.shape_grad (i, q_index)*tangent)
								    * (fe_values.shape_grad (j, q_index)*tangent)
								  +G_fiber*(normal_proj*fe_values.shape_grad(i, q_index))
								    * (normal_proj*fe_values.shape_grad(j, q_index)))
								* fe_values.JxW(q_index);

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
//			fiber_rhs,
//			fiber_rhs);

//	dh.begin_active()->get_dof_indices(local_dof_indices);
//	for (unsigned int i=0; i<dofs_per_cell; i++)
//		fiber_matrix.set(local_dof_indices[i], local_dof_indices[i],1e12);

}





void FiberLinear::output_results(const std::string &base_path, const Vector<double> &solution_3d, const Vector<double> &solution_1d) const
{
	std::string filename = base_path + "-fiber-displacement.vtk";
	std::ofstream output (filename.c_str());

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	DataOut<1,DoFHandler<1,3> > data_out;
	Vector<double> dx(dh.n_dofs()), dy(dh.n_dofs()), dz(dh.n_dofs());
	std::vector<unsigned int> count_x(dh.n_dofs(), 0.), count_y(dh.n_dofs(), 0.), count_z(dh.n_dofs(), 0.);

	data_out.attach_dof_handler(dh);

//	DoFHandler<1,3>::active_cell_iterator cell = dh.begin_active(),
//			end_cell = dh.end();
//	for (; cell!=end_cell; ++cell)
//	{
//		cell->get_dof_indices (local_dof_indices);
//		Point<3> tangent = cell->vertex(1) - cell->vertex(0);
//		tangent /= tangent.norm();
//		for (unsigned int i=0; i<dofs_per_cell; i++)
//		{
//			int ci = fe.system_to_component_index(i).first;
//			int bi = fe.system_to_component_index(i).second;
//			if (bi < 2)
//			{
//				switch (ci)
//				{
//				case 0:
//					dx[local_dof_indices[bi]] += solution_1d[local_dof_indices[i]];
//					count_x[local_dof_indices[bi]]++;
//					break;
//				case 1:
//					dy[local_dof_indices[bi]] += solution_1d[local_dof_indices[i]];
//					break;
//				case 2:
//					dz[local_dof_indices[bi]] += solution_1d[local_dof_indices[i]];
//					break;
//				}
//			}
//		}
//	}
//	for (unsigned int i=0; i<dx.size(); i++)
//	{
//		dx[i] /= count[i];
//		dy[i] /= count[i];
//		dz[i] /= count[i];
//	}

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

//	data_out.add_data_vector (solution_1d, "solution");
	data_out.add_data_vector (solution_1d, "displacement");
//	data_out.add_data_vector (dy, "dy");
//	data_out.add_data_vector (dz, "dz");
	data_out.build_patches ();
	data_out.write_vtk (output);
}








































FiberTimoshenko::FiberTimoshenko(const Parameters::AllParameters &params)
	: fe(FE_Q<1,3>(2), 3, FE_Q<1,3>(1), 3),
	  dh(tri),
	  q_constraint(
//			  QGauss<1>(1)
			  QGaussLobatto<1>(2)
			  ),
	  parameters(params)

{
	GridIn<1,3> grid_in;
	grid_in.attach_triangulation(tri);

	std::ifstream input_file(params.mesh1d_file);
	Assert (input_file, ExcFileNotOpen(params.mesh1d_file.c_str()));

	std::cout << "* Read mesh file '" << params.mesh1d_file.c_str() << "'" << std::endl;
	grid_in.read_msh(input_file);

	std::cout << "   Number of fiber active cells:       "
			<< tri.n_active_cells()
			<< std::endl;


	dh.distribute_dofs(fe);
	solution.reinit (dh.n_dofs());
}

FiberTimoshenko::~FiberTimoshenko()
{
}

void FiberTimoshenko::attach_matrix_handler(const DoFHandler<3> &dh_)
{
	dh_3d = &dh_;

	MappingQ1<3> map;
	FEValues<1,3> fe_values(fe, q_constraint, update_quadrature_points);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		fe_values.reinit(line);
		for (unsigned int k=0; k<q_constraint.size(); k++)
			node_to_cell.push_back(GridTools::find_active_cell_around_point(map, *dh_3d, fe_values.quadrature_point(k)));
	}
}

unsigned int FiberTimoshenko::get_constraint_matrix_n_rows()
{
	return 3*tri.n_lines();
}

void FiberTimoshenko::set_sparsity_pattern(SparsityPattern &sp)
{
	CompressedSparsityPattern c_sparsity(dh.n_dofs());
	DoFTools::make_sparsity_pattern(dh, c_sparsity);
	sp.copy_from(c_sparsity);
}

void FiberTimoshenko::set_constraint_matrix_sparsity_pattern(SparsityPattern &sp0t, SparsityPattern &sp1t, SparsityPattern &sp0, SparsityPattern &sp1)
{
	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), dof_indices(fe.dofs_per_cell);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		// allocate space for 3d dofs adajcent to line vertices
		for (unsigned int k=0; k<q_constraint.size(); k++)
		{
			auto cell = node_to_cell[q_constraint.size()*id+k].first;
			cell->get_dof_indices(dof_indices_3d);
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
//				if (fe_3d->system_to_component_index(i).first > 1) continue;
				sp0.add(3*id+fe_3d->system_to_component_index(i).first, dof_indices_3d[i]);
				sp0t.add(dof_indices_3d[i], 3*id+fe_3d->system_to_component_index(i).first);
			}
		}

		// allocate space for 1d dofs
		line->get_dof_indices(dof_indices);
		for (unsigned int i=0; i<fe.dofs_per_cell; i++)
		{
			// consider only displacements in 1d (ignore rotations)
			if (fe.system_to_component_index(i).first < 3)
			{
				sp1.add(3*id+fe.system_to_component_index(i).first,   dof_indices[i]);
				sp1t.add(dof_indices[i], 3*id+fe.system_to_component_index(i).first);
			}
		}
	}
}



void FiberTimoshenko::assemble_constraint_mat(SparseMatrix<double> &cm0t, SparseMatrix<double> &cm1t, SparseMatrix<double> &cm0, SparseMatrix<double> &cm1)
{
	const FiniteElement<3> *fe_3d = &dh_3d->get_fe();
	const unsigned int dofs_per_cell_3d = fe_3d->dofs_per_cell;
	std::vector<types::global_dof_index> dof_indices_3d(dofs_per_cell_3d), dof_indices(fe.dofs_per_cell);
	FEValues<1,3> fe_values1d(fe, q_constraint, update_values | update_JxW_values);

	DoFHandler<1,3>::active_cell_iterator line = dh.begin_active(),
			end_line = dh.end();
	for (unsigned int id=0; line!=end_line; ++line, ++id)
	{
		line->get_dof_indices(dof_indices);
		Point<3> tangent = line->vertex(1)-line->vertex(0);
		tangent /= tangent.norm();

		fe_values1d.reinit(line);

		for (unsigned int k=0; k<q_constraint.size(); k++)
		{
			auto cell = node_to_cell[q_constraint.size()*id+k].first;
			Quadrature<3> q(node_to_cell[q_constraint.size()*id+k].second);
			FEValues<3> fe_values3d(*fe_3d, q, update_values);

			fe_values3d.reinit(cell);
			cell->get_dof_indices(dof_indices_3d);
			for (unsigned int i=0; i<dofs_per_cell_3d; i++)
			{
				cm0.add(3*id+fe_3d->system_to_component_index(i).first, dof_indices_3d[i],
						fe_values3d.shape_value(i,0)*fe_values1d.JxW(k));
				cm0t.add(dof_indices_3d[i], 3*id+fe_3d->system_to_component_index(i).first,
						fe_values3d.shape_value(i,0)*fe_values1d.JxW(k));
			}

			for (unsigned int i=0; i<fe.dofs_per_cell; i++)
			{
				// consider only displacements (ignore rotations)
				if (fe.system_to_component_index(i).first < 3)
				{
					cm1.add(3*id+fe.system_to_component_index(i).first, dof_indices[i], -fe_values1d.shape_value(i,k)*fe_values1d.JxW(k));
					cm1t.add(dof_indices[i], 3*id+fe.system_to_component_index(i).first, -fe_values1d.shape_value(i,k)*fe_values1d.JxW(k));
				}
			}
		}
	}
}




void FiberTimoshenko::assemble_fiber_matrix(SparseMatrix<double> &fiber_matrix, Vector<double> &fiber_rhs, double E_fiber, double nu_fiber, double fiber_volume)
{
	QGauss<1> quadrature_formula(8);
	FEValues<1,3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();
	FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	Tensor<2,3> Pmat, Qmat, Imat, gu_i, gu_j, gth_i, gth_j;
	Tensor<1,3> th_i, th_j, cross_th_i, cross_th_j;
//	double J = fiber_volume*fiber_volume/acos(-1)*0.5;
	double J = fiber_volume*fiber_volume/acos(-1);
	double G_fiber = 0.5 * E_fiber / (nu_fiber + 1);

	printf("   Fiber parameters:\n    Young_modulus = %g\n    Shear_modulus = %g\n\n", E_fiber, G_fiber);

	DoFHandler<1,3>::active_cell_iterator cell = dh.begin_active(),
			end_cell = dh.end();
	for (; cell!=end_cell; ++cell)
	{
		fe_values.reinit(cell);
		cell_matrix = 0;
		Point<3> tangent = cell->vertex(1) - cell->vertex(0);
		tangent /= tangent.norm();
		for (unsigned int i=0; i<3; i++)
			for (unsigned int j=0; j<3; j++)
			{
				Pmat[{i,j}] = tangent[i]*tangent[j];
				Qmat[{i,j}] = (i==j?1.:0.) - tangent[i]*tangent[j];
			}
//		Imat = fiber_volume*fiber_volume/acos(-1)*(0.5*Pmat + 0.25*Qmat);
		Imat = fiber_volume*fiber_volume/acos(-1)*0.5*Qmat;

		for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
		{
			for (unsigned int i=0; i<dofs_per_cell; i++)
			{
				gu_i = 0;
				th_i = 0;
				gth_i = 0;
				int ci = fe.system_to_component_index(i).first;
				if (ci < 3)
					gu_i[ci] = fe_values.shape_grad(i, q_index);
				else
				{
					th_i[ci-3] = fe_values.shape_value(i, q_index);
					gth_i[ci-3] = fe_values.shape_grad(i, q_index);
					cross_product(cross_th_i, th_i, tangent);
				}

				for (unsigned int j=0; j<dofs_per_cell; j++)
				{
					gu_j = 0;
					th_j = 0;
					gth_j = 0;
					int cj = fe.system_to_component_index(j).first;
					if (cj < 3)
						gu_j[cj] = fe_values.shape_grad(j, q_index);
					else
					{
						th_j[cj-3] = fe_values.shape_value(j, q_index);
						gth_j[cj-3] = fe_values.shape_grad(j, q_index);
						cross_product(cross_th_j, th_j, tangent);
					}

					cell_matrix(i,j) += (
							 fiber_volume*E_fiber*(Pmat*(gu_i*tangent))*(Pmat*(gu_j*tangent))
							+fiber_volume*G_fiber*(Qmat*(gu_i*tangent) - cross_th_i)*(Qmat*(gu_j*tangent) - cross_th_j)
							+E_fiber*(Imat*(gth_i*tangent))*(gth_j*tangent)
							+G_fiber*J*(Pmat*(gth_i*tangent))*(Pmat*(gth_j*tangent))
							)*fe_values.JxW(q_index);
				}
			}

		}

		cell->get_dof_indices (local_dof_indices);
		for (unsigned int i=0; i<dofs_per_cell; ++i)
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				fiber_matrix.add (local_dof_indices[i], local_dof_indices[j], -cell_matrix(i,j));

	}

	// apply boundary conditions
	std::map<types::global_dof_index,double> boundary_values;
	FEValuesExtractors::Vector displacements(0);
	ComponentMask displacement_mask = fe.component_mask(displacements);
	for (auto bc : parameters.bc)
	{
		FunctionParser<3> fp(3);
		fp.initialize("x,y,z", bc.second, {});

		VectorTools::interpolate_boundary_values (dh,
				bc.first,
				fp,
				boundary_values,
				displacement_mask);
	}
	MatrixTools::apply_boundary_values (boundary_values,
			fiber_matrix,
			solution,
			fiber_rhs);
}





void FiberTimoshenko::output_results(const std::string &base_path, const Vector<double> &solution_3d, const Vector<double> &solution_1d) const
{
	std::string filename = base_path + "-fiber-displacement.vtk";
	std::ofstream output (filename.c_str());

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	DataOut<1,DoFHandler<1,3> > data_out;
	Vector<double> dx(dh.n_dofs()), dy(dh.n_dofs()), dz(dh.n_dofs());
	std::vector<unsigned int> count_x(dh.n_dofs(), 0.), count_y(dh.n_dofs(), 0.), count_z(dh.n_dofs(), 0.);

	std::vector<std::string> solution_names(3, "displacement");
	solution_names.insert(solution_names.end(), 3, "angle");
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation(6, DataComponentInterpretation::component_is_part_of_vector);

	data_out.attach_dof_handler(dh);

	data_out.add_data_vector(solution_1d,
			solution_names,
			DataOut<1,DoFHandler<1,3> >::type_dof_data,
			data_component_interpretation);

	data_out.build_patches();
	data_out.write_vtk(output);
}

