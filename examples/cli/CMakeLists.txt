add_executable(ced-cli main.cpp)
target_link_libraries(ced-cli PRIVATE ced)
target_include_directories(ced-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party)
