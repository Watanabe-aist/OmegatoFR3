////////////////////////////////////////////////////////////////////////////////
//
//  This example implements a simple spring model which pulls the device
//  towards the center of the workspace. If the user presses the user button,
//  the application exits.
//
////////////////////////////////////////////////////////////////////////////////
// C++ library headers
#include <iomanip>
#include <iostream>
// project headers
#include "dhdc.h"
// constants
constexpr double K = 1000.0;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
  // open a connection to the device
  if (dhdOpen() < 0) {
    std::cout << "error: cannot open device" << std::endl;
    return -1;
  }
  // run haptic loop
  int done = 0;
  while (!done) {
    // get end-effector position
    double px, py, pz;
    dhdGetPosition(&px, &py, &pz);
    std::cout << px << ", " << py << ", " << pz << std::endl;
  }
  // close the connection
  dhdClose();
  return 0;
}