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

#include "elastic.hh"


using namespace Composite_elasticity_problem;
using namespace dealii;







template <int dim>
ElasticProblem<dim>::ElasticProblem(const Parameters::AllParameters &params)
:
dof_handler (triangulation),
fe (FE_Q<dim>(1), dim),
parameters(params)
{
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);

	std::ifstream input_file(parameters.mesh_file);
	Assert (input_file, ExcFileNotOpen(parameters.mesh_file.c_str()));

	std::cout << "* Read mesh file '" << parameters.mesh_file << "'"<< std::endl;
	grid_in.read_msh(input_file);

	std::cout << "   Number of active cells:             "
			<< triangulation.n_active_cells()
			<< std::endl;

	dof_handler.distribute_dofs(fe);
}



template <int dim>
ElasticProblem<dim>::~ElasticProblem ()
{
	dof_handler.clear ();
}


template <int dim>
void ElasticProblem<dim>::setup_system ()
{
	solution.reinit (dof_handler.n_dofs());

	std::cout << "   Number of degrees of freedom: "
			  << dof_handler.n_dofs()
			  << "\n\n";
}

template<int dim>
Tensor<4,dim> ElasticProblem<dim>::elastic_tensor(unsigned int material_id) const
{
	Tensor<4,dim> tensor;
	double lambda, mu;

	static const double Young_modulus_matrix_fiber
			= parameters.Fiber_volume_ratio*parameters.Young_modulus_fiber
			 +(1.-parameters.Fiber_volume_ratio)*parameters.Young_modulus_matrix;

	static const double Poisson_ratio_matrix_fiber
			= parameters.Fiber_volume_ratio*parameters.Poisson_ratio_fiber
			 +(1.-parameters.Fiber_volume_ratio)*parameters.Poisson_ratio_matrix;

	if (material_id == parameters.Reinforcement_material_id)
	{
		// fibre
		lambda = Young_modulus_matrix_fiber*Poisson_ratio_matrix_fiber/((1.+Poisson_ratio_matrix_fiber)*(1.-2*Poisson_ratio_matrix_fiber));
		mu     = Young_modulus_matrix_fiber*0.5/(1.+Poisson_ratio_matrix_fiber);
	}
	else
	{
		// matrix
		lambda = parameters.Young_modulus_matrix*parameters.Poisson_ratio_matrix/((1.+parameters.Poisson_ratio_matrix)*(1.-2*parameters.Poisson_ratio_matrix));
		mu     = parameters.Young_modulus_matrix*0.5/(1.+parameters.Poisson_ratio_matrix);
	}

	for (int i=0; i<dim; i++)
		for (int j=0; j<dim; j++)
			tensor[i][j][i][j] += mu;

	for (int i=0; i<dim; i++)
		for (int j=0; j<dim; j++)
			tensor[i][i][j][j] += lambda;

	return tensor;
}


template <int dim>
void ElasticProblem<dim>::assemble_system(SparseMatrix<double> &system_matrix, Vector<double> &system_rhs)
{
	QGauss<dim>  quadrature_formula(2);

	FEValues<dim> fe_values (fe, quadrature_formula,
			update_values   | update_gradients |
			update_quadrature_points | update_JxW_values);

	QGauss<dim-1> quadrature_face(2);

	FESystem<dim-1> fe_face(FE_Q<dim-1>(1), dim);

	FEFaceValues<dim> fe_face_values(fe, quadrature_face, update_values | update_quadrature_points | update_JxW_values);

	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();

	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	std::vector<double>     lambda_values (n_q_points);
	std::vector<double>     mu_values (n_q_points);

	Tensor<4,dim> el_tensor;

	std::vector<Vector<double> > rhs_values (n_q_points, Vector<double>(dim)),
			trac_values (quadrature_face.size(), Vector<double>(dim));

	double Young_modulus_matrix_fiber = parameters.Fiber_volume_ratio*parameters.Young_modulus_fiber + (1.-parameters.Fiber_volume_ratio)*parameters.Young_modulus_matrix;
	double Poisson_ratio_matrix_fiber = parameters.Fiber_volume_ratio*parameters.Poisson_ratio_fiber + (1.-parameters.Fiber_volume_ratio)*parameters.Poisson_ratio_matrix;

	printf("   Matrix parameters:\n    Young_modulus = %g\n    Poisson_ratio = %g\n\n", parameters.Young_modulus_matrix, parameters.Poisson_ratio_matrix);
	if (!parameters.use_1d_fibers)
		printf("   Reinforcement parameters:\n    Young_modulus = %g\n    Poisson_ratio = %g\n\n", Young_modulus_matrix_fiber, Poisson_ratio_matrix_fiber);


	// Now we can begin with the loop over all cells:
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		cell_matrix = 0;
		cell_rhs = 0;

		fe_values.reinit (cell);

		// Next we get the values of the coefficients at the quadrature
		// points. Likewise for the right hand side:
		el_tensor = elastic_tensor(cell->material_id());

		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			const unsigned int
			component_i = fe.system_to_component_index(i).first;

			for (unsigned int j=0; j<dofs_per_cell; ++j)
			{
				const unsigned int
				component_j = fe.system_to_component_index(j).first;

				for (unsigned int k=0; k<dim; k++)
					for (unsigned int l=0; l<dim; l++)
					{
						double el_sum = 0.25*(
								el_tensor[k][component_i][l][component_j] +
								el_tensor[component_i][k][l][component_j] +
								el_tensor[k][component_i][component_j][l] +
								el_tensor[component_i][k][component_j][l]
							);

						for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
						{

									cell_matrix(i,j) += el_sum
														*fe_values.shape_grad(j,q_point)[l]*fe_values.shape_grad(i,q_point)[k]
														*fe_values.JxW(q_point);
		//					cell_matrix(i,j) +=
		//                    				(
		//                    						(fe_values.shape_grad(i,q_point)[component_i] *
		//                    								fe_values.shape_grad(j,q_point)[component_j] *
		//                    								lambda_values[q_point])
		//                    						+
		//											(fe_values.shape_grad(i,q_point)[component_j] *
		//													fe_values.shape_grad(j,q_point)[component_i] *
		//													mu_values[q_point])
		//											+
		//											((component_i == component_j) ?
		//													(fe_values.shape_grad(i,q_point) *
		//															fe_values.shape_grad(j,q_point) *
		//															mu_values[q_point])  : 0)
		//                    				)
		//                    				*
		//                    				fe_values.JxW(q_point);
						}
					}
			}
		}

		// Assembling the right hand side due to volume force
		if (parameters.force.find(cell->material_id()) != parameters.force.end())
		{
			FunctionParser<dim> fp(dim);
			fp.initialize("x,y,z", parameters.force[cell->material_id()], {});
			fp.vector_value_list (fe_values.get_quadrature_points(), rhs_values);
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				const unsigned int
				component_i = fe.system_to_component_index(i).first;

				for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
					cell_rhs(i) += fe_values.shape_value(i,q_point) *
					rhs_values[q_point](component_i) *
					fe_values.JxW(q_point);
			}
		}

		// Assembling rhs due to traction at boundary
		for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
		{
			if (cell->at_boundary(face))
			{
				if (parameters.traction.find(cell->face(face)->boundary_indicator()) == parameters.traction.end())
					continue;

				fe_face_values.reinit(cell, face);

				FunctionParser<dim> fp(dim);
				fp.initialize("x,y,z", parameters.traction[cell->face(face)->boundary_indicator()], {});
				fp.vector_value_list (fe_face_values.get_quadrature_points(), trac_values);

				for (unsigned int i=0; i<fe.dofs_per_cell; i++)
				{
					const unsigned int component_i = fe.system_to_component_index(i).first;

					for (unsigned int q_point=0; q_point<quadrature_face.size(); q_point++)
					{
						cell_rhs(i) += fe_face_values.shape_value(i,q_point) *
								trac_values[q_point](component_i) *
								fe_face_values.JxW(q_point);
					}

				}
			}
		}

		// The transfer from local degrees of freedom into the global matrix
		// and right hand side vector does not depend on the equation under
		// consideration, and is thus the same as in all previous
		// examples. The same holds for the elimination of hanging nodes from
		// the matrix and right hand side, once we are done with assembling
		// the entire linear system:
		cell->get_dof_indices (local_dof_indices);
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				system_matrix.add (local_dof_indices[i],
						local_dof_indices[j],
						cell_matrix(i,j));

			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}







	// The interpolation of the boundary values needs a small modification:
	// since the solution function is vector-valued, so need to be the
	// boundary values. The <code>ZeroFunction</code> constructor accepts a
	// parameter that tells it that it shall represent a vector valued,
	// constant zero function with that many components. By default, this
	// parameter is equal to one, in which case the <code>ZeroFunction</code>
	// object would represent a scalar function. Since the solution vector has
	// <code>dim</code> components, we need to pass <code>dim</code> as number
	// of components to the zero function as well.
	std::map<types::global_dof_index,double> boundary_values;
	for (auto bc : parameters.bc)
	{
		FunctionParser<dim> fp(dim);
		fp.initialize("x,y,z", bc.second, {});

		VectorTools::interpolate_boundary_values (dof_handler,
				bc.first,
				fp,
				boundary_values);
	}
	MatrixTools::apply_boundary_values (boundary_values,
			system_matrix,
			solution,
			system_rhs,
			false);
}

template<int dim>
void ElasticProblem<dim>::create_output_vectors() const
{
	Vector<double> stress_vector, energy_vector, material_vector;
	DoFHandler<dim> elem_handler (triangulation);
	FE_DGP<dim> fe_stress(0);

	QGauss<dim>  quadrature_formula(1);

	FEValues<dim> fe_values (fe, quadrature_formula,
			update_values   | update_gradients |
			update_quadrature_points | update_JxW_values);

	std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
	std::vector<types::global_dof_index> elem_local_dof_indices (fe_stress.dofs_per_cell);

	elem_handler.distribute_dofs(fe_stress);
	stress_vector.reinit(elem_handler.n_dofs());
	energy_vector.reinit(elem_handler.n_dofs());
	material_vector.reinit(elem_handler.n_dofs());


	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end();
	typename DoFHandler<dim>::active_cell_iterator elem_cell = elem_handler.begin_active();
	for (; cell!=endc; ++cell, ++elem_cell)
	{
		fe_values.reinit (cell);

		cell->get_dof_indices (local_dof_indices);
		elem_cell->get_dof_indices (elem_local_dof_indices);

		Tensor<4,dim> el_tensor = elastic_tensor(cell->material_id());

		Tensor<2,dim> sym_grad;

		for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
		{
			const unsigned int component_i = fe.system_to_component_index(i).first;

			for (unsigned int k=0; k<dim; k++)
			{
				sym_grad[component_i][k] += 0.5*fe_values.shape_grad(i,0)[k] * solution[local_dof_indices[i]];
				sym_grad[k][component_i] += 0.5*fe_values.shape_grad(i,0)[k] * solution[local_dof_indices[i]];
			}
		}

		Tensor<2,dim> stress;
		for (unsigned int i=0; i<dim; i++)
			for (unsigned int j=0; j<dim; j++)
				for (unsigned int k=0; k<dim; k++)
					for (unsigned int l=0; l<dim; l++)
				stress[i][j] = el_tensor[i][j][k][l] * sym_grad[k][l];

		stress_vector[elem_local_dof_indices[0]] = sqrt(1.5*stress.norm_square() - 0.5*pow(trace(stress),2.));
		energy_vector[elem_local_dof_indices[0]] = scalar_product(stress, sym_grad);
		material_vector[elem_local_dof_indices[0]] = cell->material_id();

	}

	std::string filename = parameters.output_file_base + "-stress.vtk";
	std::ofstream output (filename.c_str());

	DataOut<dim> data_out;
	data_out.attach_dof_handler (elem_handler);

	data_out.add_data_vector (stress_vector, "vonMisesStress", DataOut_DoFData<DoFHandler<dim>,dim>::type_cell_data);
	data_out.add_data_vector (energy_vector, "energy", DataOut_DoFData<DoFHandler<dim>,dim>::type_cell_data);
	data_out.add_data_vector (material_vector, "material", DataOut_DoFData<DoFHandler<dim>,dim>::type_cell_data);
	data_out.build_patches ();
	data_out.write_vtk (output);

	elem_handler.clear();

}



template<int dim>
void ElasticProblem<dim>::output_ranges() const
{
	const unsigned int n_base_dofs = fe.base_element(0).dofs_per_cell;
	double min_val[dim], max_val[dim], max_norm2 = 0, *norm2;
	std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);

	norm2 = new double[n_base_dofs];

	for (int i=0; i<dim; i++)
	{
		min_val[i] = std::numeric_limits<double>::max();
		max_val[i] = -std::numeric_limits<double>::max();
	}

	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int j=0; j<n_base_dofs; j++) norm2[j] = 0;

		cell->get_dof_indices (local_dof_indices);

		for (unsigned int i=0; i<fe.dofs_per_cell; i++)
		{
			int comp = fe.system_to_component_index(i).first;
			norm2[fe.system_to_component_index(i).second] += solution[local_dof_indices[i]]*solution[local_dof_indices[i]];

			if (solution[local_dof_indices[i]] > max_val[comp]) max_val[comp] = solution[local_dof_indices[i]];
			if (solution[local_dof_indices[i]] < min_val[comp]) min_val[comp] = solution[local_dof_indices[i]];
		}

		for (unsigned int j=0; j<n_base_dofs; j++)
			if (norm2[j] > max_norm2)
				max_norm2 = norm2[j];
	}

	delete[] norm2;

	printf("   Solution ranges:\n");
	for (int i=0; i<dim; i++)
		printf("    component %d: [%g, %g]\n", i, min_val[i], max_val[i]);
	printf("    norm: %g\n\n", sqrt(max_norm2));
}


template <int dim>
void ElasticProblem<dim>::output_results () const
{
	std::string filename = parameters.output_file_base + "-matrix.vtk";
	std::ofstream output (filename.c_str());

	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);



	std::vector<std::string> solution_names;
	std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
	for (int i=0; i<dim; i++)
	{
		solution_names.push_back ("displacement");
		interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
	}

	data_out.add_data_vector (solution,
			solution_names,
			DataOut_DoFData<DoFHandler<dim>,dim>::type_automatic,
			interpretation);



	// calculate additional output fields (energy, stress, material_id)
	Vector<double> stress_vector(triangulation.n_cells()), energy_vector(triangulation.n_cells()), material_vector(triangulation.n_cells());
	QGauss<dim>  quadrature_formula(1);
	FEValues<dim> fe_values (fe, quadrature_formula,
			update_values   | update_gradients |
			update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end();
	for (unsigned int cell_index = 0; cell!=endc; ++cell_index, ++cell)
	{
		fe_values.reinit (cell);
		cell->get_dof_indices (local_dof_indices);
		Tensor<4,dim> el_tensor = elastic_tensor(cell->material_id());
		Tensor<2,dim> sym_grad, stress;

		for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
		{
			const unsigned int component_i = fe.system_to_component_index(i).first;

			for (unsigned int k=0; k<dim; k++)
			{
				sym_grad[component_i][k] += 0.5*fe_values.shape_grad(i,0)[k] * solution[local_dof_indices[i]];
				sym_grad[k][component_i] += 0.5*fe_values.shape_grad(i,0)[k] * solution[local_dof_indices[i]];
			}
		}

		for (unsigned int i=0; i<dim; i++)
			for (unsigned int j=0; j<dim; j++)
				for (unsigned int k=0; k<dim; k++)
					for (unsigned int l=0; l<dim; l++)
						stress[i][j] = el_tensor[i][j][k][l] * sym_grad[k][l];

		stress_vector[cell_index] = sqrt(1.5*stress.norm_square() - 0.5*pow(trace(stress),2.));
		energy_vector[cell_index] = scalar_product(stress, sym_grad);
		material_vector[cell_index] = cell->material_id();
	}
	data_out.add_data_vector(stress_vector, "vonMisesStress", DataOut_DoFData<DoFHandler<dim>,dim>::type_cell_data);
	data_out.add_data_vector(energy_vector, "energy", DataOut_DoFData<DoFHandler<dim>,dim>::type_cell_data);
	data_out.add_data_vector(material_vector, "material", DataOut_DoFData<DoFHandler<dim>,dim>::type_cell_data);




	data_out.build_patches ();
	data_out.write_vtk (output);



	output_ranges();
}












template class ElasticProblem<3>;






