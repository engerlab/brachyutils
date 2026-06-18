cmake_minimum_required(VERSION 3.16.3)
project(brachyutils)

# In the Slicer build environment, PYTHON_EXECUTABLE is automatically 
# set to Slicer's internal Python interpreter.
# We create a custom target that runs Slicer's pip during the build process.

add_custom_target(InstallBrachyUtils ALL
  # Install the python package
  COMMAND ${PYTHON_EXECUTABLE} -m pip install --disable-pip-version-check --no-warn-script-location ${CMAKE_CURRENT_SOURCE_DIR}
  
  # Copy the non-python 'admin' data folder into the site-packages directory
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/admin" "${CMAKE_BINARY_DIR}/../python-install/Lib/site-packages/admin"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Installing local brachyutils and its dependencies into Slicer's Python environment..."
)