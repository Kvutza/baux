cmake_minimum_required(VERSION 3.26)
project(baux_iree LANGUAGES C CXX OBJC)
add_subdirectory("${IREE_SOURCE}" iree EXCLUDE_FROM_ALL)
add_library(baux_iree SHARED bridge.m)
target_link_libraries(baux_iree PRIVATE iree::runtime::unified)
set_target_properties(baux_iree PROPERTIES
    INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_NAME_DIR TRUE)

target_compile_options(baux_iree PRIVATE -fobjc-arc)
