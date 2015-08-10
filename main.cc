#include <iostream>
#include "coupled.hh"











int main (int argc, char **argv)
{
  try
    {
      dealii::deallog.depth_console (0);

      std::cout << "============================================" << std::endl
    		  	<< "comp_el - FE simulation of linear elasticity in fiber-reinforced composites" << std::endl
				<< "============================================" << std::endl << std::endl
				<< "* Read parameters from '" << argv[1] << "'" << std::endl;

      Composite_elasticity_problem::CoupledProblem problem(argv[1]);
      problem.run ();

      std::cout << "Done." << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
