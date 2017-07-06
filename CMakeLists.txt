cmake_minimum_required(VERSION 2.8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(libcubamm)
set(libcubamm_VERSION_MAJOR 0)
set(libcubamm_VERSION_MINOR 0)

include(CheckIncludeFiles)
check_include_files("cuba.h" HAVE_CUBA_H)
if(NOT HAVE_CUBA_H)
  unset(HAVE_CUBA_H CACHE)
  message(FATAL_ERROR "header cuba.h not found")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++14 -Wno-missing-braces")

file(GLOB src_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB tst_files "${CMAKE_CURRENT_SOURCE_DIR}/t/*.cpp")
set(inc_dir "${CMAKE_CURRENT_SOURCE_DIR}/include/")
file(GLOB header_files "${inc_dir}/*/*.hpp")

include_directories(${inc_dir})

add_library(cubamm SHARED ${src_files})
target_link_libraries (cubamm ${req_libraries})

add_library(cubamm_static STATIC ${src_files})
target_link_libraries (cubamm_static ${req_libraries})
set_target_properties(cubamm_static PROPERTIES OUTPUT_NAME cubamm)

enable_testing()
foreach(tst ${tst_files})
  get_filename_component(tst_exe ${tst} NAME_WE)
  add_executable(${tst_exe} ${tst})
  target_link_libraries (${tst_exe} ${req_libraries} cubamm)
  add_test("t_${tst_exe}" ${tst_exe})
endforeach()

install(TARGETS cubamm cubamm_static DESTINATION lib)
install(FILES ${header_files} DESTINATION include/cubamm/)

# build a CPack driven installer package
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_PACKAGE_VERSION_MAJOR "${cubamm_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cubamm_VERSION_MINOR}")
set(CPACK_SOURCE_IGNORE_FILES ".out$" "~$" "/build/" "/doc/" ".gitignore")
include(CPack)
