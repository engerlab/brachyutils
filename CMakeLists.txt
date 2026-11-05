cmake_minimum_required(VERSION 3.16.3)
project(brachyutils)

# In the Slicer build environment, PYTHON_EXECUTABLE is automatically 
# set to Slicer's internal Python interpreter.
# We create a custom target that runs Slicer's pip during the build process.
# Linux only: use CPU-only torch (PyPI's default on Linux is the large CUDA build).
# On Windows and macOS this does nothing.
if(UNIX AND NOT APPLE)
  set(TORCH_INDEX_ARGS --index-url https://download.pytorch.org/whl/cpu)
else()
  set(TORCH_INDEX_ARGS "")
endif()

# Where the 'admin' folder gets copied, which differs between Windows and Linux
if(WIN32)
  set(ADMIN_DEST "${CMAKE_BINARY_DIR}/../python-install/Lib/site-packages/admin")
else()
  set(ADMIN_DEST "${CMAKE_BINARY_DIR}/../python-install/lib/python3.12/site-packages/admin")
endif()

set(PIP_ARGS --disable-pip-version-check --no-warn-script-location)

add_custom_target(InstallBrachyUtils ALL
  # 1. torch first, so later steps reuse it
  COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PIP_ARGS} torch==2.15.0 ${TORCH_INDEX_ARGS}

  # 2. botorch, Ax, optuna and the rest of what brachyutils imports
  COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PIP_ARGS} -r ${CMAKE_CURRENT_SOURCE_DIR}/slicer-requirements.txt

  # 3. brachyutils itself
  COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PIP_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}

  # 4. Copy the non-python 'admin' data folder into site-packages
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/admin" "${ADMIN_DEST}"

  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Installing local brachyutils and its dependencies into Slicer's Python environment..."
  VERBATIM
)